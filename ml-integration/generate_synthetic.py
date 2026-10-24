from __future__ import annotations

"""Generate clearly labeled synthetic telemetry and evaluate research-only models.

Generated rows mirror the ThermalCollector training CSV contract. They are not
real sensor data and are deliberately stored outside the live model registry.
"""

import csv
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "synthetic-lab"
CSV_PATH = OUTPUT / "synthetic_thermal_data.csv"
ARTIFACT_DIR = OUTPUT / "artifacts"
FIELDS = [
    "schemaVersion", "sessionId", "timestamp", "cpuTemperatureC",
    "gpuTemperatureC", "cpuUtilizationPercent", "cpuClockMhz", "cpuPowerW",
    "cpuFanSpeedRpm", "gpuUtilizationPercent", "memoryUtilizationPercent",
    "gpuSensor",
]


def _bounded(value: float, low: float, high: float) -> float:
    return round(min(high, max(low, value)), 2)


def generate() -> int:
    rng = random.Random(20261023)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    duration = 60 * 60
    dt = 5
    # Most sessions remain low/moderate load, reflecting observed utilization;
    # a smaller share intentionally covers ramp, sustained-load, and recovery.
    scenario_names = ["idle", "idle", "normal", "idle", "ramp", "normal",
                      "idle", "mixed", "cooling", "normal", "sustained", "idle"]
    for session_no in range(12):
        session = f"synthetic-session-{session_no + 1:02d}"
        scenario = scenario_names[session_no % len(scenario_names)]
        session_start = start + timedelta(hours=session_no)
        ambient = 22 + (session_no % 4) * 1.2
        cpu_temp = 40.5 + rng.uniform(-1.5, 2.0)
        gpu_temp = 39.5 + rng.uniform(-1.0, 2.0)
        cpu_load = rng.uniform(5, 18)
        gpu_load = rng.uniform(0, 8)
        for tick in range(duration // dt):
            elapsed = tick * dt
            phase = elapsed / duration
            if scenario == "idle":
                cpu_target, gpu_target = 10 + 4 * math.sin(elapsed / 110), 3
            elif scenario == "ramp":
                cpu_target = 12 + 75 * min(1, phase * 2.2)
                gpu_target = 5 + 42 * min(1, phase * 1.8)
            elif scenario == "sustained":
                cpu_target = 78 + 8 * math.sin(elapsed / 95)
                gpu_target = 52 + 12 * math.sin(elapsed / 130)
            elif scenario == "cooling":
                cpu_target = 82 - 68 * min(1, max(0, (phase - .25) / .65))
                gpu_target = 56 - 48 * min(1, max(0, (phase - .25) / .65))
            elif scenario == "mixed":
                cpu_target = 18 + (58 if int(elapsed // 180) % 2 else 12)
                gpu_target = 8 + (50 if int(elapsed // 240) % 2 else 5)
            else:
                cpu_target = 28 + 22 * math.sin(elapsed / 150) + 8 * math.sin(elapsed / 31)
                gpu_target = 12 + 15 * math.sin(elapsed / 170)
            cpu_load += (cpu_target - cpu_load) * 0.16 + rng.gauss(0, 2.1)
            gpu_load += (gpu_target - gpu_load) * 0.18 + rng.gauss(0, 2.5)
            cpu_load = _bounded(cpu_load, 1, 100)
            gpu_load = _bounded(gpu_load, 0, 100)
            cpu_power = 8 + 0.72 * cpu_load + rng.gauss(0, 2.0)
            cpu_clock = 1800 + 19 * cpu_load + rng.gauss(0, 90)
            fan = 800 + max(0, cpu_temp - 45) * 48 + rng.gauss(0, 90)
            cpu_equilibrium = ambient + 19 + 0.43 * cpu_load + 0.08 * gpu_load
            gpu_equilibrium = ambient + 14 + 0.34 * gpu_load + 0.05 * cpu_load
            cpu_temp += (cpu_equilibrium - cpu_temp) * dt / 210 + rng.gauss(0, .12)
            gpu_temp += (gpu_equilibrium - gpu_temp) * dt / 260 + rng.gauss(0, .10)
            cpu_temp = _bounded(cpu_temp, 32, 92)
            gpu_temp = _bounded(gpu_temp, 30, 88)
            memory = _bounded(40 + 0.43 * cpu_load + rng.gauss(0, 3.5), 20, 96)
            rows.append({
                "schemaVersion": 1,
                "sessionId": session,
                "timestamp": (session_start + timedelta(seconds=elapsed)).isoformat().replace("+00:00", "Z"),
                "cpuTemperatureC": cpu_temp,
                "gpuTemperatureC": gpu_temp,
                "cpuUtilizationPercent": cpu_load,
                "cpuClockMhz": _bounded(cpu_clock, 800, 5200),
                "cpuPowerW": _bounded(cpu_power, 3, 150),
                "cpuFanSpeedRpm": _bounded(fan, 0, 7000),
                "gpuUtilizationPercent": gpu_load,
                "memoryUtilizationPercent": memory,
                "gpuSensor": "SYNTHETIC",
            })
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def evaluate() -> dict[str, object]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from smarttherm_ml.dataset import load_points, build_examples, discover_csv_files
    from smarttherm_ml.training import split_chronologically, _features_matrix, _metrics
    from smarttherm_ml.features import FEATURE_NAMES

    points, rejected = load_points([CSV_PATH])
    real_points, _ = load_points(discover_csv_files(ROOT))
    examples, target_rejected = build_examples(points)
    component = "CPU"
    items = examples[component]
    train, validation, test = split_chronologically(items)
    x_train, x_val, x_test = map(_features_matrix, (train, validation, test))
    y_train = np.asarray([x.target_c for x in train])
    y_val = np.asarray([x.target_c for x in validation])
    y_test = np.asarray([x.target_c for x in test])
    candidates = {
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", GradientBoostingRegressor(loss="huber", n_estimators=150, learning_rate=.04, max_depth=3, min_samples_leaf=5, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", XGBRegressor(objective="reg:squarederror", n_estimators=250, max_depth=4, learning_rate=.04, min_child_weight=5, subsample=.85, colsample_bytree=.9, reg_lambda=2, tree_method="hist", n_jobs=2, random_state=42)),
        ]),
    }
    baseline_val = np.asarray([x.persistence_prediction_c for x in validation])
    baseline_test = np.asarray([x.persistence_prediction_c for x in test])
    def distribution(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "median": None, "max": None}
        return {"count": len(values), "min": round(float(np.min(values)), 2),
                "median": round(float(np.median(values)), 2), "max": round(float(np.max(values)), 2)}

    columns: dict[str, list[float]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        schema_matches = reader.fieldnames == FIELDS
        for row in reader:
            for column in ("cpuTemperatureC", "gpuTemperatureC", "cpuUtilizationPercent", "cpuClockMhz", "gpuUtilizationPercent", "memoryUtilizationPercent"):
                columns.setdefault(column, []).append(float(row[column]))
    source_fields = {
        "cpuTemperatureC": "cpu_temperature_c", "gpuTemperatureC": "gpu_temperature_c",
        "cpuUtilizationPercent": "cpu_utilization_percent", "gpuUtilizationPercent": "gpu_utilization_percent",
        "memoryUtilizationPercent": "memory_utilization_percent", "cpuClockMhz": "cpu_clock_mhz",
    }
    distributions = {name: {
        "real": distribution([getattr(p, field) for p in real_points if getattr(p, field) is not None]),
        "synthetic": distribution(columns.get(name, [])),
    } for name, field in source_fields.items()}

    def corr(data: list, left: str, right: str) -> float | None:
        pairs = [(getattr(p, left), getattr(p, right)) for p in data
                 if getattr(p, left) is not None and getattr(p, right) is not None]
        if len(pairs) < 3:
            return None
        value = float(np.corrcoef(np.asarray(pairs, dtype=float).T)[0, 1])
        return round(value, 3) if math.isfinite(value) else None

    intervals = [(b.timestamp-a.timestamp).total_seconds() for a, b in zip(points, points[1:]) if a.session_id == b.session_id]
    cpu_steps = [abs(b.cpu_temperature_c-a.cpu_temperature_c) for a, b in zip(points, points[1:])
                 if a.session_id == b.session_id and a.cpu_temperature_c is not None and b.cpu_temperature_c is not None]
    results: dict[str, object] = {
        "dataKind": "SYNTHETIC_ONLY_NOT_REAL_WORLD_VALIDATION",
        "csv": str(CSV_PATH.relative_to(ROOT)),
        "rows": len(points),
        "cpuExamples": len(items),
        "rejectedRows": dict(rejected),
        "rejectedTargets": dict(target_rejected),
        "splitCounts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "horizonSeconds": 600,
        "schema": {"matchesCollectorTrainingCsv": schema_matches, "columns": FIELDS,
                   "timestampFormat": "ISO-8601 UTC with Z suffix", "samplingSeconds": 5,
                   "missingValues": "empty cells; none generated", "temperatureUnit": "C",
                   "utilizationUnit": "percent", "clockUnit": "MHz", "powerUnit": "W", "fanUnit": "RPM"},
        "quality": {"duplicateSessionTimestamps": len(points)-len({(p.session_id,p.timestamp) for p in points}),
                    "schemaMatches": schema_matches, "invalidRows": dict(rejected),
                    "samplingIntervalSeconds": distribution(intervals),
                    "absoluteCpuTemperatureStepC": distribution(cpu_steps)},
        "featureDistributions": distributions,
        "correlations": {
            "real_cpu_temp_vs_cpu_utilization": corr(real_points, "cpu_temperature_c", "cpu_utilization_percent"),
            "synthetic_cpu_temp_vs_cpu_utilization": corr(points, "cpu_temperature_c", "cpu_utilization_percent"),
            "real_gpu_temp_vs_gpu_utilization": corr(real_points, "gpu_temperature_c", "gpu_utilization_percent"),
            "synthetic_gpu_temp_vs_gpu_utilization": corr(points, "gpu_temperature_c", "gpu_utilization_percent"),
        },
        "correlationCaveat": "Real CPU temperature correlation uses only 6 readings and is not statistically meaningful; clock frequency is absent from all parsed real records.",
        "featureNames": FEATURE_NAMES,
        "baseline": {
            "validation": {**_metrics(y_val, baseline_val), "r2": round(float(r2_score(y_val, baseline_val)), 4)},
            "test": {**_metrics(y_test, baseline_test), "r2": round(float(r2_score(y_test, baseline_test)), 4)},
        },
        "candidates": {},
    }
    if not schema_matches or intervals and any(interval != 5 for interval in intervals):
        raise ValueError("Synthetic CSV schema or sampling interval validation failed")
    fitted: dict[str, tuple[Pipeline, float]] = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        val_metrics = _metrics(y_val, model.predict(x_val))
        test_metrics = _metrics(y_test, model.predict(x_test))
        val_metrics["r2"] = round(float(r2_score(y_val, model.predict(x_val))), 4)
        test_metrics["r2"] = round(float(r2_score(y_test, model.predict(x_test))), 4)
        results["candidates"][name] = {"validation": val_metrics, "test": test_metrics}
        fitted[name] = (model, val_metrics["rmse_c"])
    selected = min(fitted, key=lambda name: fitted[name][1])
    model = fitted[selected][0]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "synthetic_cpu_model.joblib"
    joblib.dump({
        "schemaVersion": 1,
        "component": "CPU",
        "model": selected,
        "dataKind": "SYNTHETIC_ONLY_NOT_FOR_LIVE_PREDICTIONS",
        "horizonSeconds": 600,
        "featureNames": FEATURE_NAMES,
        "trainingEnvironment": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "metrics": results,
        "pipeline": model,
    }, artifact_path)
    results["selectedSyntheticCandidate"] = selected
    results["selectionBasis"] = "validation RMSE; test split remains held out"
    chosen_test = results["candidates"][selected]["test"]
    baseline_test_metrics = results["baseline"]["test"]
    beats_synthetic_test_baseline = (
        chosen_test["mae_c"] < baseline_test_metrics["mae_c"]
        and chosen_test["rmse_c"] < baseline_test_metrics["rmse_c"]
    )
    results["syntheticHoldoutBaselineGate"] = "PASS" if beats_synthetic_test_baseline else "FAIL"
    results["artifact"] = str(artifact_path.relative_to(ROOT))
    results["livePromotion"] = "BLOCKED: synthetic data cannot validate or activate a live thermal model"
    (OUTPUT / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    count = generate()
    print(f"Generated {count} explicitly synthetic telemetry rows: {CSV_PATH}")
    print(json.dumps(evaluate(), indent=2))
