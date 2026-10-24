from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np

from .dataset import SensorPoint

FEATURE_NAMES = [
    "temperature_current_c",
    "temperature_mean_30s_c",
    "temperature_mean_60s_c",
    "temperature_mean_300s_c",
    "temperature_min_60s_c",
    "temperature_max_60s_c",
    "temperature_slope_30s_c_per_min",
    "temperature_slope_300s_c_per_min",
    "cpu_utilization_percent",
    "cpu_utilization_mean_60s_percent",
    "cpu_clock_mhz",
    "cpu_clock_mean_60s_mhz",
    "gpu_utilization_percent",
    "gpu_utilization_mean_60s_percent",
    "memory_utilization_percent",
]


def compute_features(
    history: Iterable[SensorPoint],
    component: str,
) -> dict[str, float | None]:
    if component not in {"CPU", "GPU"}:
        raise ValueError("component must be CPU or GPU")
    ordered = sorted(history, key=lambda point: point.timestamp)
    if not ordered:
        raise ValueError("sensor history is empty")
    current_time = ordered[-1].timestamp
    temp_attribute = "cpu_temperature_c" if component == "CPU" else "gpu_temperature_c"

    def values(attribute: str, window_seconds: int) -> list[tuple[datetime, float]]:
        start = current_time - timedelta(seconds=window_seconds)
        result = [
            (point.timestamp, value)
            for point in ordered
            if point.timestamp >= start
            and point.timestamp <= current_time
            and (value := getattr(point, attribute)) is not None
            and math.isfinite(value)
        ]
        return result

    temp_30 = values(temp_attribute, 30)
    temp_60 = values(temp_attribute, 60)
    temp_300 = values(temp_attribute, 300)
    if not temp_30 or temp_30[-1][0] != current_time:
        raise ValueError("latest component temperature is unavailable")

    cpu_60 = values("cpu_utilization_percent", 60)
    clock_60 = values("cpu_clock_mhz", 60)
    gpu_60 = values("gpu_utilization_percent", 60)

    def latest(attribute: str) -> float | None:
        value = getattr(ordered[-1], attribute)
        return float(value) if value is not None and math.isfinite(value) else None

    def mean(items: list[tuple[datetime, float]]) -> float | None:
        return float(np.mean([item[1] for item in items])) if items else None

    return {
        "temperature_current_c": latest(temp_attribute),
        "temperature_mean_30s_c": mean(temp_30),
        "temperature_mean_60s_c": mean(temp_60),
        "temperature_mean_300s_c": mean(temp_300),
        "temperature_min_60s_c": min((item[1] for item in temp_60), default=None),
        "temperature_max_60s_c": max((item[1] for item in temp_60), default=None),
        "temperature_slope_30s_c_per_min": _slope_per_minute(temp_30),
        "temperature_slope_300s_c_per_min": _slope_per_minute(temp_300),
        "cpu_utilization_percent": latest("cpu_utilization_percent"),
        "cpu_utilization_mean_60s_percent": mean(cpu_60),
        "cpu_clock_mhz": latest("cpu_clock_mhz"),
        "cpu_clock_mean_60s_mhz": mean(clock_60),
        "gpu_utilization_percent": latest("gpu_utilization_percent"),
        "gpu_utilization_mean_60s_percent": mean(gpu_60),
        "memory_utilization_percent": latest("memory_utilization_percent"),
    }


def _slope_per_minute(items: list[tuple[datetime, float]]) -> float | None:
    if len(items) < 3:
        return None
    x = np.array([(timestamp - items[0][0]).total_seconds() for timestamp, _ in items])
    y = np.array([value for _, value in items])
    if x[-1] < 10:
        return None
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0:
        return None
    slope_per_second = float(np.dot(centered_x, y - y.mean()) / denominator)
    return slope_per_second * 60

