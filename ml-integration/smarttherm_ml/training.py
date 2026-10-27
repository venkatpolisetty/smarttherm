from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .dataset import (
    TARGET_HORIZON_SECONDS,
    TrainingExample,
    build_examples,
    discover_csv_files,
    load_points,
)
from .features import FEATURE_NAMES

logger = logging.getLogger("smarttherm.ml.training")
MIN_TRAINING_EXAMPLES = 300
MIN_EVALUATION_EXAMPLES = 50
TIME_BLOCK_SECONDS = 60 * 60
MODEL_SCHEMA_VERSION = 2


class InsufficientTrainingDataError(ValueError):
    pass


class _PersistenceResidualPipeline:
    def __init__(self, residual_model: Pipeline, correction_weight: float) -> None:
        self.residual_model = residual_model
        self.correction_weight = correction_weight

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("features do not match the active model contract")
        current_temperature_index = FEATURE_NAMES.index("temperature_current_c")
        current_temperature = values[:, current_temperature_index]
        return current_temperature + self.correction_weight * self.residual_model.predict(values)


def assess_examples(points: list[Any]) -> dict[str, Any]:
    examples, rejected = build_examples(points)
    valid_sessions = {
        example.session_id for component_examples in examples.values()
        for example in component_examples
    }
    segments = _continuous_segments(points)
    report: dict[str, Any] = {
        "sensorRows": len(points),
        "sensorSessionCount": len({point.session_id for point in points}),
        "sessionsWithValidTargets": len(valid_sessions),
        "continuousSegments": len(segments),
        "featureCoverage": {
            field: sum(getattr(point, field) is not None for point in points)
            for field in (
                "cpu_temperature_c",
                "gpu_temperature_c",
                "cpu_utilization_percent",
                "cpu_clock_mhz",
                "gpu_utilization_percent",
                "memory_utilization_percent",
            )
        },
        "targetHorizonSeconds": TARGET_HORIZON_SECONDS,
        "targetToleranceSeconds": 15,
        "maxGapSeconds": 20,
        "components": {},
        "rejected": dict(rejected),
    }
    for component, items in examples.items():
        report["components"][component] = _example_summary(items)
    return report


def _continuous_segments(points: list[Any]) -> list[list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for point in points:
        if point.source_available:
            grouped.setdefault(point.session_id, []).append(point)
    segments: list[list[Any]] = []
    for session_points in grouped.values():
        session_points.sort(key=lambda item: item.timestamp)
        segment: list[Any] = []
        for point in session_points:
            if segment and (point.timestamp - segment[-1].timestamp).total_seconds() > 20:
                segments.append(segment)
                segment = []
            segment.append(point)
        if segment:
            segments.append(segment)
    return segments


def split_chronologically(
    examples: list[TrainingExample],
    minimum_train: int = MIN_TRAINING_EXAMPLES,
    minimum_evaluation: int = MIN_EVALUATION_EXAMPLES,
    *,
    enforce_minimums: bool = True,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    ordered = sorted(examples, key=lambda item: item.timestamp)
    if enforce_minimums and len(ordered) < minimum_train + 2 * minimum_evaluation:
        raise InsufficientTrainingDataError(
            f"Need at least {minimum_train + 2 * minimum_evaluation} examples "
            f"({minimum_train} train and {minimum_evaluation} each validation/test); "
            f"found {len(ordered)}."
        )

    blocks: dict[tuple[str, int], list[TrainingExample]] = {}
    for item in ordered:
        block = int(item.timestamp.timestamp() // TIME_BLOCK_SECONDS)
        blocks.setdefault((item.session_id, block), []).append(item)
    ordered_blocks = sorted(
        blocks.items(),
        key=lambda item: min(example.timestamp for example in item[1]),
    )
    if len(ordered_blocks) < 3 and enforce_minimums:
        raise InsufficientTrainingDataError(
            "Need at least three distinct one-hour session blocks for chronological "
            "train, validation, and test partitions."
        )
    if len(ordered_blocks) < 3:
        return [], [], []
    first_block_index = max(1, int(len(ordered_blocks) * 0.70))
    second_block_index = max(first_block_index + 1, int(len(ordered_blocks) * 0.85))
    second_block_index = min(second_block_index, len(ordered_blocks) - 1)
    first_boundary = min(
        example.timestamp for example in ordered_blocks[first_block_index][1]
    )
    second_boundary = min(
        example.timestamp for example in ordered_blocks[second_block_index][1]
    )
    train = [
        item for item in ordered
        if item.timestamp + timedelta(seconds=TARGET_HORIZON_SECONDS) < first_boundary
    ]
    validation = [
        item for item in ordered
        if first_boundary <= item.timestamp
        and item.timestamp + timedelta(seconds=TARGET_HORIZON_SECONDS) < second_boundary
    ]
    test = [item for item in ordered if item.timestamp >= second_boundary]

    if enforce_minimums and (
        len(train) < minimum_train
        or len(validation) < minimum_evaluation
        or len(test) < minimum_evaluation
    ):
        raise InsufficientTrainingDataError(
            "After chronological splitting and the five-minute target embargo, "
            f"found train={len(train)}, validation={len(validation)}, test={len(test)}; "
            f"require at least {minimum_train}, {minimum_evaluation}, and {minimum_evaluation}."
        )
    return train, validation, test


def train_component(
    component: str,
    examples: list[TrainingExample],
    model_dir: Path,
    minimum_train: int = MIN_TRAINING_EXAMPLES,
    minimum_evaluation: int = MIN_EVALUATION_EXAMPLES,
) -> dict[str, Any]:
    train, validation, test = split_chronologically(
        examples, minimum_train, minimum_evaluation
    )
    incumbent, manifest = _load_incumbent(component, model_dir)
    x_train = _features_matrix(train)
    y_train = np.asarray([item.target_c for item in train], dtype=float)
    x_validation = _features_matrix(validation)
    y_validation = np.asarray([item.target_c for item in validation], dtype=float)
    x_test = _features_matrix(test)
    y_test = np.asarray([item.target_c for item in test], dtype=float)

    candidates: dict[str, tuple[Pipeline, bool]] = {
        "XGBoostDirect": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    (
                        "regressor",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=250,
                            max_depth=4,
                            learning_rate=0.04,
                            min_child_weight=5,
                            subsample=0.85,
                            colsample_bytree=0.9,
                            reg_lambda=2.0,
                            tree_method="hist",
                            n_jobs=2,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            False,
        ),
        "GradientBoosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    (
                        "regressor",
                        GradientBoostingRegressor(
                            loss="huber",
                            n_estimators=150,
                            learning_rate=0.04,
                            max_depth=3,
                            min_samples_leaf=5,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            False,
        ),
        "XGBoostResidual": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    (
                        "regressor",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=250,
                            max_depth=4,
                            learning_rate=0.04,
                            min_child_weight=5,
                            subsample=0.85,
                            colsample_bytree=0.9,
                            reg_lambda=2.0,
                            tree_method="hist",
                            n_jobs=2,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            True,
        ),
    }
    validation_baseline = np.asarray(
        [item.persistence_prediction_c for item in validation], dtype=float
    )
    baseline_validation_metrics = _metrics(y_validation, validation_baseline)
    fitted: dict[str, tuple[Any, dict[str, float]]] = {}
    current_temperature_index = FEATURE_NAMES.index("temperature_current_c")
    current_train = x_train[:, current_temperature_index]
    current_validation = x_validation[:, current_temperature_index]
    for name, (model, predicts_residual) in candidates.items():
        target = y_train - current_train if predicts_residual else y_train
        model.fit(x_train, target)
        if predicts_residual:
            residual_validation = model.predict(x_validation)
            correction_weight = _select_residual_correction_weight(
                y_validation,
                validation_baseline,
                residual_validation,
            )
            validation_prediction = (
                current_validation + correction_weight * residual_validation
            )
            prediction_model = _PersistenceResidualPipeline(model, correction_weight)
        else:
            validation_prediction = model.predict(x_validation)
            prediction_model = model
        fitted[name] = (
            prediction_model,
            _metrics(y_validation, validation_prediction),
        )

    validation_eligible = [
        name for name, (_, metrics) in fitted.items()
        if metrics["mae_c"] < baseline_validation_metrics["mae_c"]
        and metrics["rmse_c"] < baseline_validation_metrics["rmse_c"]
    ]
    if not validation_eligible:
        return {
            "component": component,
            "status": "NOT_PROMOTED",
            "reason": "Neither candidate improved on the persistence baseline on validation data.",
            "examples": _split_counts(train, validation, test),
            "validationBaseline": baseline_validation_metrics,
            "candidates": {
                name: {"validation": metrics} for name, (_, metrics) in fitted.items()
            },
        }

    selected = min(
        validation_eligible,
        key=lambda name: fitted[name][1]["rmse_c"],
    )
    selected_model, selected_validation_metrics = fitted[selected]
    test_metrics = _metrics(y_test, selected_model.predict(x_test))
    test_baseline = np.asarray(
        [item.persistence_prediction_c for item in test], dtype=float
    )
    test_baseline_metrics = _metrics(y_test, test_baseline)
    report: dict[str, Any] = {
        "component": component,
        "status": "NOT_PROMOTED",
        "reason": "Selected model did not improve on both held-out test metrics.",
        "candidate": selected,
        "examples": _split_counts(train, validation, test),
        "validationBaseline": baseline_validation_metrics,
        "validation": selected_validation_metrics,
        "testBaseline": test_baseline_metrics,
        "test": test_metrics,
        "candidates": {
            name: {"validation": metrics} for name, (_, metrics) in fitted.items()
        },
    }
    if not _beats_baseline(test_metrics, test_baseline_metrics):
        return report
    if incumbent is not None:
        incumbent_metrics = _metrics(y_test, incumbent["pipeline"].predict(x_test))
        report["incumbent"] = {
            "model": incumbent["model"],
            "test": incumbent_metrics,
        }
        if not _does_not_regress(test_metrics, incumbent_metrics):
            report["reason"] = "Candidate would regress the active model on held-out test data."
            return report

    # Keep the validation and test metrics tied to models that never saw those
    # partitions. Once promotion is earned, refit on all available history so
    # the installed model benefits from every collected example.
    all_features = _features_matrix(examples)
    all_targets = np.asarray([item.target_c for item in examples], dtype=float)
    if isinstance(selected_model, _PersistenceResidualPipeline):
        residual_model = clone(selected_model.residual_model)
        all_current = all_features[:, current_temperature_index]
        residual_model.fit(all_features, all_targets - all_current)
        installed_model = _PersistenceResidualPipeline(
            residual_model,
            selected_model.correction_weight,
        )
    else:
        installed_model = clone(selected_model)
        installed_model.fit(all_features, all_targets)

    artifact_dir = model_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    timestamps = [item.timestamp for item in examples]
    artifact_name = (
        f"{component.lower()}-{selected.lower()}-"
        f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}.joblib"
    )
    artifact_path = artifact_dir / artifact_name
    artifact = {
        "schemaVersion": MODEL_SCHEMA_VERSION,
        "targetHorizonSeconds": TARGET_HORIZON_SECONDS,
        "component": component,
        "model": selected,
        "trainedAt": created_at.isoformat(),
        "dataStart": min(timestamps).isoformat(),
        "dataEnd": max(timestamps).isoformat(),
        "trainingExamples": len(examples),
        "fitScope": "all_eligible_history_after_chronological_holdout_evaluation",
        "trainingEnvironment": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "featureNames": FEATURE_NAMES,
        "metrics": {
            "validation": selected_validation_metrics,
            "test": test_metrics,
            "testBaseline": test_baseline_metrics,
            "examples": _split_counts(train, validation, test),
        },
        "pipeline": installed_model,
    }
    _atomic_joblib_dump(artifact, artifact_path)
    manifest_path = artifact_dir / "manifest.json"
    manifest = manifest or {"schemaVersion": MODEL_SCHEMA_VERSION, "models": {}}
    manifest["schemaVersion"] = MODEL_SCHEMA_VERSION
    manifest.setdefault("models", {})
    manifest["models"][component] = {
        "artifact": artifact_name,
        "model": selected,
        "trainedAt": created_at.isoformat(),
        "dataStart": min(timestamps).isoformat(),
        "dataEnd": max(timestamps).isoformat(),
        "trainingExamples": len(examples),
        "fitScope": "all_eligible_history_after_chronological_holdout_evaluation",
        "trainingEnvironment": artifact["trainingEnvironment"],
        "featureNames": FEATURE_NAMES,
        "metrics": artifact["metrics"],
    }
    _atomic_json_dump(manifest, manifest_path)
    report["status"] = "PROMOTED"
    report["reason"] = (
        "Improved on the persistence baseline on validation and held-out test data "
        "without regressing the active model."
    )
    report["artifact"] = str(artifact_path)
    _prune_component_artifacts(artifact_dir, component, artifact_name)
    return report


def train_from_history(
    project_root: Path,
    model_dir: Path,
    minimum_train: int = MIN_TRAINING_EXAMPLES,
    minimum_evaluation: int = MIN_EVALUATION_EXAMPLES,
) -> dict[str, Any]:
    with _training_process_lock(model_dir) as acquired:
        if not acquired:
            return {
                "status": "TRAINING_BUSY",
                "csvFilesInspected": 0,
                "validSensorRows": 0,
                "rejectedSensorRows": {},
                "rejectedTargetCandidates": {},
                "components": {
                    component: {
                        "component": component,
                        "status": "TRAINING_BUSY",
                        "reason": "Another training job already holds the model-directory lock.",
                    }
                    for component in ("CPU", "GPU")
                },
            }
        return _run_training_cycle(
            project_root,
            model_dir,
            minimum_train,
            minimum_evaluation,
        )


def _run_training_cycle(
    project_root: Path,
    model_dir: Path,
    minimum_train: int,
    minimum_evaluation: int,
) -> dict[str, Any]:
    files = discover_csv_files(project_root)
    points, rejected_rows = load_points(files)
    examples, rejected_targets = build_examples(points)
    components: dict[str, dict[str, Any]] = {}
    for component in ("CPU", "GPU"):
        try:
            components[component] = train_component(
                component,
                examples[component],
                model_dir,
                minimum_train,
                minimum_evaluation,
            )
        except InsufficientTrainingDataError as exception:
            components[component] = {
                "component": component,
                "status": "INSUFFICIENT_DATA",
                "reason": str(exception),
                "validFiveMinutePairs": len(examples[component]),
            }
        except Exception as exception:
            logger.exception("Training failed for %s.", component)
            components[component] = {
                "component": component,
                "status": "TRAINING_ERROR",
                "reason": str(exception),
                "validFiveMinutePairs": len(examples[component]),
            }
    statuses = [item["status"] for item in components.values()]
    status = (
        "PROMOTED" if "PROMOTED" in statuses
        else "INSUFFICIENT_DATA" if all(item == "INSUFFICIENT_DATA" for item in statuses)
        else "NOT_PROMOTED" if "NOT_PROMOTED" in statuses
        else "TRAINING_BUSY" if all(item == "TRAINING_BUSY" for item in statuses)
        else "TRAINING_ERROR"
    )
    return {
        "status": status,
        "csvFilesInspected": len(files),
        "validSensorRows": len(points),
        "rejectedSensorRows": dict(rejected_rows),
        "rejectedTargetCandidates": dict(rejected_targets),
        "components": components,
    }


@contextmanager
def _training_process_lock(model_dir: Path) -> Iterator[bool]:
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / ".training.lock"
    with lock_path.open("a+b") as handle:
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    acquired = False
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError:
                    acquired = False
            yield acquired
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _beats_baseline(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        candidate["mae_c"] < baseline["mae_c"]
        and candidate["rmse_c"] < baseline["rmse_c"]
    )


def _does_not_regress(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    return (
        candidate["mae_c"] <= incumbent["mae_c"]
        and candidate["rmse_c"] <= incumbent["rmse_c"]
        and (
            candidate["mae_c"] < incumbent["mae_c"]
            or candidate["rmse_c"] < incumbent["rmse_c"]
        )
    )


def _load_incumbent(
    component: str,
    model_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ValueError("The active model manifest is unreadable; refusing model promotion.") from exception
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), dict):
        raise ValueError("The active model manifest is invalid; refusing model promotion.")
    if manifest.get("schemaVersion") not in {1, MODEL_SCHEMA_VERSION}:
        raise ValueError("The active model manifest version is unsupported; refusing model promotion.")
    entry = manifest["models"].get(component)
    if entry is None:
        return None, manifest
    if not isinstance(entry, dict):
        raise ValueError(f"The active {component} model entry is invalid; refusing promotion.")
    filename = entry.get("artifact")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"The active {component} model artifact path is invalid; refusing promotion.")
    artifact_path = (model_dir / filename).resolve()
    if artifact_path.parent != model_dir.resolve() or not artifact_path.is_file():
        raise ValueError(f"The active {component} model artifact is missing; refusing promotion.")
    try:
        incumbent = joblib.load(artifact_path)
    except Exception as exception:
        raise ValueError(f"The active {component} model artifact cannot be loaded.") from exception
    if isinstance(incumbent, dict) and incumbent.get("schemaVersion") == 1:
        # Version 1 artifacts forecast ten minutes and are not comparable to
        # the five-minute candidate being trained. Replace on successful promotion.
        return None, manifest
    if (
        not isinstance(incumbent, dict)
        or incumbent.get("schemaVersion") != MODEL_SCHEMA_VERSION
        or incumbent.get("targetHorizonSeconds") != TARGET_HORIZON_SECONDS
        or incumbent.get("component") != component
        or incumbent.get("featureNames") != FEATURE_NAMES
        or not hasattr(incumbent.get("pipeline"), "predict")
    ):
        raise ValueError(f"The active {component} model artifact is invalid; refusing promotion.")
    return incumbent, manifest


def _prune_component_artifacts(
    model_dir: Path,
    component: str,
    active_artifact: str,
    keep: int = 3,
) -> None:
    artifacts = sorted(
        model_dir.glob(f"{component.lower()}-*.joblib"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    retained = {path.name for path in artifacts[:keep]}
    retained.add(active_artifact)
    for artifact in artifacts:
        if artifact.name not in retained:
            try:
                artifact.unlink()
            except OSError as exception:
                logger.warning("Could not remove superseded model artifact %s: %s", artifact, exception)


def _features_matrix(examples: list[TrainingExample]) -> np.ndarray:
    return np.asarray(
        [
            [item.features.get(name) if item.features.get(name) is not None else np.nan
             for name in FEATURE_NAMES]
            for item in examples
        ],
        dtype=float,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    differences = predicted - actual
    return {
        "mae_c": round(float(np.mean(np.abs(differences))), 4),
        "rmse_c": round(float(math.sqrt(np.mean(differences**2))), 4),
    }


def _select_residual_correction_weight(
    actual: np.ndarray,
    persistence: np.ndarray,
    predicted_residual: np.ndarray,
) -> float:
    baseline = _metrics(actual, persistence)
    mae_scale = max(baseline["mae_c"], 1e-12)
    rmse_scale = max(baseline["rmse_c"], 1e-12)
    weights = np.linspace(0.0, 1.0, 101)

    def score(weight: float) -> tuple[float, float, float]:
        metrics = _metrics(actual, persistence + weight * predicted_residual)
        return (
            metrics["mae_c"] / mae_scale + metrics["rmse_c"] / rmse_scale,
            metrics["rmse_c"],
            metrics["mae_c"],
        )

    return float(min(weights, key=score))


def _split_counts(
    train: list[TrainingExample],
    validation: list[TrainingExample],
    test: list[TrainingExample],
) -> dict[str, int]:
    return {"train": len(train), "validation": len(validation), "test": len(test)}


def _example_summary(examples: list[TrainingExample]) -> dict[str, Any]:
    train, validation, test = split_chronologically(
        examples,
        enforce_minimums=False,
    )
    split_counts = _split_counts(train, validation, test)
    if len(examples) < MIN_TRAINING_EXAMPLES + 2 * MIN_EVALUATION_EXAMPLES:
        reason = (
            f"Need at least {MIN_TRAINING_EXAMPLES + 2 * MIN_EVALUATION_EXAMPLES} examples "
            f"({MIN_TRAINING_EXAMPLES} train and {MIN_EVALUATION_EXAMPLES} each "
            f"validation/test); found {len(examples)}."
        )
    elif len({
        (item.session_id, int(item.timestamp.timestamp() // TIME_BLOCK_SECONDS))
        for item in examples
    }) < 3:
        reason = (
            "Need at least three distinct one-hour session blocks for chronological "
            "train, validation, and test partitions."
        )
    elif (
        len(train) < MIN_TRAINING_EXAMPLES
        or len(validation) < MIN_EVALUATION_EXAMPLES
        or len(test) < MIN_EVALUATION_EXAMPLES
    ):
        reason = (
            "After chronological splitting and the five-minute target embargo, "
            f"found train={len(train)}, validation={len(validation)}, test={len(test)}; "
            f"require at least {MIN_TRAINING_EXAMPLES}, {MIN_EVALUATION_EXAMPLES}, "
            f"and {MIN_EVALUATION_EXAMPLES}."
        )
    else:
        reason = None
    summary: dict[str, Any] = {
        "validFiveMinutePairs": len(examples),
        "splitCounts": split_counts,
        "trainingStatus": "INSUFFICIENT_DATA" if reason else "SUFFICIENT_DATA",
        "trainingReason": reason,
    }
    if not examples:
        return summary
    summary.update({
        "sessions": len({item.session_id for item in examples}),
        "firstTargetTimestamp": min(item.timestamp for item in examples).isoformat(),
        "lastTargetTimestamp": max(item.timestamp for item in examples).isoformat(),
    })
    return summary


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
