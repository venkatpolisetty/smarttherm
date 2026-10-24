from __future__ import annotations

from bisect import bisect_left
import csv
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MAX_GAP_SECONDS = 20
TARGET_HORIZON_SECONDS = 10 * 60
TARGET_TOLERANCE_SECONDS = 15
MIN_FEATURE_SPAN_SECONDS = 10
MAX_TEMPERATURE_C = 150.0


@dataclass(frozen=True)
class SensorPoint:
    timestamp: datetime
    session_id: str
    cpu_temperature_c: float | None
    gpu_temperature_c: float | None
    cpu_utilization_percent: float | None
    cpu_clock_mhz: float | None
    gpu_utilization_percent: float | None
    memory_utilization_percent: float | None
    gpu_sensor: str | None = None
    source_available: bool = True


@dataclass(frozen=True)
class TrainingExample:
    timestamp: datetime
    session_id: str
    component: str
    features: dict[str, float | None]
    target_c: float
    persistence_prediction_c: float


def discover_csv_files(project_root: Path) -> list[Path]:
    root = project_root.resolve()
    directories = (root, root / "ThermalCollector", root / "ThermalCollector" / "data")
    found: set[Path] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for pattern in ("telemetry*.csv", "training_telemetry_*.csv"):
            found.update(
                path.resolve()
                for path in directory.glob(pattern)
                if path.is_file() and not path.name.startswith("process_")
            )
    return sorted(found)


def load_points(paths: Iterable[Path]) -> tuple[list[SensorPoint], Counter[str]]:
    points: list[SensorPoint] = []
    rejected: Counter[str] = Counter()
    for path in sorted({Path(item).resolve() for item in paths}):
        if not path.is_file() or not path.name.lower().endswith(".csv"):
            continue
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as source:
                for row in csv.DictReader(source):
                    point = _point_from_row(row, path, rejected)
                    if point is not None:
                        points.append(point)
        except (OSError, csv.Error) as exc:
            rejected["file_read_error"] += 1
            print(f"Could not read {path}: {exc}")

    points.sort(key=lambda item: (item.session_id, item.timestamp))
    unique: list[SensorPoint] = []
    seen: set[tuple[str, datetime]] = set()
    for point in points:
        key = (point.session_id, point.timestamp)
        if key in seen:
            rejected["duplicate_timestamp"] += 1
            continue
        seen.add(key)
        unique.append(point)
    return unique, rejected


def build_examples(
    points: Iterable[SensorPoint],
) -> tuple[dict[str, list[TrainingExample]], Counter[str]]:
    grouped: dict[str, list[SensorPoint]] = {}
    for point in points:
        if not point.source_available:
            continue
        grouped.setdefault(point.session_id, []).append(point)
    examples: dict[str, list[TrainingExample]] = {"CPU": [], "GPU": []}
    rejected: Counter[str] = Counter()

    for session_id, session_points in grouped.items():
        session_points.sort(key=lambda item: item.timestamp)
        segment: list[SensorPoint] = []
        for point in session_points:
            if segment and (point.timestamp - segment[-1].timestamp).total_seconds() > MAX_GAP_SECONDS:
                _build_segment_examples(segment, examples, rejected)
                segment = []
            segment.append(point)
        if segment:
            _build_segment_examples(segment, examples, rejected)
    for component in examples:
        examples[component].sort(key=lambda item: item.timestamp)
    return examples, rejected


def _build_segment_examples(
    points: list[SensorPoint],
    examples: dict[str, list[TrainingExample]],
    rejected: Counter[str],
) -> None:
    from .features import compute_features

    timestamps = [point.timestamp.timestamp() for point in points]
    for index, point in enumerate(points):
        target_time = timestamps[index] + TARGET_HORIZON_SECONDS
        insertion_point = bisect_left(timestamps, target_time, lo=index + 1)
        candidates = [
            candidate
            for candidate in (insertion_point - 1, insertion_point)
            if index < candidate < len(timestamps)
        ]
        target_index = min(candidates, key=lambda candidate: abs(timestamps[candidate] - target_time), default=-1)
        if target_index < 0 or abs(timestamps[target_index] - target_time) > TARGET_TOLERANCE_SECONDS:
            rejected["missing_ten_minute_target"] += 1
            continue

        history = points[max(0, index - 120) : index + 1]
        if len(history) < 3 or (history[-1].timestamp - history[0].timestamp).total_seconds() < MIN_FEATURE_SPAN_SECONDS:
            rejected["insufficient_feature_history"] += 1
            continue

        for component, attribute in (
            ("CPU", "cpu_temperature_c"),
            ("GPU", "gpu_temperature_c"),
        ):
            current = getattr(point, attribute)
            target = getattr(points[target_index], attribute)
            if not _valid_temperature(current):
                rejected[f"{component.lower()}_missing_current_temperature"] += 1
                continue
            if not _valid_temperature(target):
                rejected[f"{component.lower()}_missing_target_temperature"] += 1
                continue
            try:
                features = compute_features(history, component)
            except ValueError:
                rejected[f"{component.lower()}_invalid_feature_history"] += 1
                continue
            examples[component].append(
                TrainingExample(
                    timestamp=point.timestamp,
                    session_id=point.session_id,
                    component=component,
                    features=features,
                    target_c=target,
                    persistence_prediction_c=current,
                )
            )


def _point_from_row(
    row: dict[str, str | None],
    path: Path,
    rejected: Counter[str],
) -> SensorPoint | None:
    raw_time = _first(row, "timestamp", "Timestamp")
    timestamp = _parse_timestamp(raw_time)
    if timestamp is None:
        rejected["invalid_timestamp"] += 1
        return None

    cpu_temp = _metric(row, ("cpuTemperatureC", "CPU_Temp"), 0, MAX_TEMPERATURE_C)
    gpu_temp = _metric(
        row,
        ("gpuTemperatureC", "NVIDIA_GPU_Temp", "Intel_GPU_Temp", "GPU_Temp"),
        0,
        MAX_TEMPERATURE_C,
    )
    cpu_util = _metric(row, ("cpuUtilizationPercent", "CPU_Load"), 0, 100)
    cpu_clock = _metric(row, ("cpuClockMhz", "CPU_Clock_MHz", "CPU_Clock"), 0, 10_000, exclusive_min=True)
    gpu_util = _metric(
        row, ("gpuUtilizationPercent", "NVIDIA_GPU_Load", "Intel_GPU_Load", "GPU_Load"), 0, 100
    )
    memory_util = _metric(row, ("memoryUtilizationPercent", "RAM_Load"), 0, 100)

    if all(value is None for value in (cpu_temp, gpu_temp, cpu_util, cpu_clock, gpu_util, memory_util)):
        rejected["row_without_valid_sensor_metrics"] += 1
        return None

    session_id = _first(row, "sessionId", "session_id") or path.name
    gpu_sensor = _first(row, "gpuSensor", "gpu_sensor") or None
    source_available = str(row.get("sourceAvailable", "true")).strip().lower() not in {
        "false",
        "0",
    }
    return SensorPoint(
        timestamp=timestamp,
        session_id=session_id,
        cpu_temperature_c=cpu_temp,
        gpu_temperature_c=gpu_temp,
        cpu_utilization_percent=cpu_util,
        cpu_clock_mhz=cpu_clock,
        gpu_utilization_percent=gpu_util,
        memory_utilization_percent=memory_util,
        gpu_sensor=gpu_sensor,
        source_available=source_available,
    )


def _first(row: dict[str, str | None], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.astimezone()
    return timestamp.astimezone(timezone.utc)


def _metric(
    row: dict[str, str | None],
    names: tuple[str, ...],
    minimum: float,
    maximum: float,
    exclusive_min: bool = False,
) -> float | None:
    for name in names:
        raw = row.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        lower_ok = value > minimum if exclusive_min else value >= minimum
        if lower_ok and value <= maximum:
            return value
        return None
    return None


def _valid_temperature(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0 <= value <= MAX_TEMPERATURE_C
