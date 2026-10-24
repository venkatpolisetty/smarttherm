from __future__ import annotations

import json
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .dataset import MAX_GAP_SECONDS, SensorPoint
from .features import FEATURE_NAMES, compute_features
from .auto_training import BackgroundTrainingCoordinator

logger = logging.getLogger("smarttherm.ml")
MAX_READINGS = 150
MAX_READING_AGE = timedelta(seconds=15)
PREDICTION_HORIZON = timedelta(minutes=10)
MAX_TEMPERATURE_C = 150
MODEL_DIR = Path(
    os.environ.get(
        "SMARTTHERM_MODEL_DIR",
        str(Path(__file__).resolve().parents[1] / "models"),
    )
).resolve()
PROJECT_ROOT = Path(
    os.environ.get("SMARTTHERM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()
# New collector data should be noticed soon after it arrives. The data and
# promotion gates still prevent training or replacing a model on every sample.
DEFAULT_TRAINING_INTERVAL_SECONDS = 15 * 60


class ReadingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime
    sessionId: str | None = None
    sourceAvailable: bool = True
    cpuTemperatureC: float | None = None
    gpuTemperatureC: float | None = None
    cpuUtilizationPercent: float | None = None
    cpuClockMhz: float | None = None
    gpuUtilizationPercent: float | None = None
    memoryUtilizationPercent: float | None = None

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator(
        "cpuTemperatureC",
        "gpuTemperatureC",
    )
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and (
            not math.isfinite(value) or value < 0 or value > MAX_TEMPERATURE_C
        ):
            raise ValueError("temperature must be finite and between 0°C and 150°C")
        return value

    @field_validator("cpuUtilizationPercent", "gpuUtilizationPercent", "memoryUtilizationPercent")
    @classmethod
    def validate_percent(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0 or value > 100):
            raise ValueError("utilization must be finite and between 0% and 100%")
        return value

    @field_validator("cpuClockMhz")
    @classmethod
    def validate_clock(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0 or value > 10_000):
            raise ValueError("clock speed must be finite and between 0 and 10,000 MHz")
        return value


class PredictionRequest(BaseModel):
    readings: list[ReadingRequest] = Field(min_length=1, max_length=MAX_READINGS)

    @field_validator("readings")
    @classmethod
    def validate_order(cls, value: list[ReadingRequest]) -> list[ReadingRequest]:
        if any(
            later.timestamp <= earlier.timestamp
            for earlier, later in zip(value, value[1:])
        ):
            raise ValueError("readings must have strictly increasing timestamps")
        return value


class ModelRegistry:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.resolve()
        self.models: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}
        self._lock = threading.RLock()
        self._reload_lock = threading.Lock()
        self._manifest_signature: tuple[int, int] | None = None
        self.reload_if_changed()

    def reload_if_changed(self) -> None:
        with self._reload_lock:
            self._reload_if_changed()

    def _reload_if_changed(self) -> None:
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.is_file():
            with self._lock:
                self.models.clear()
                self.errors.clear()
                self._manifest_signature = None
            return
        try:
            stat = manifest_path.stat()
        except OSError as exc:
            logger.error("Could not inspect model manifest %s: %s", manifest_path, exc)
            return
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if signature == self._manifest_signature:
                return
            loaded_models = dict(self.models)
            loaded_errors = dict(self.errors)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Could not read model manifest %s: %s", manifest_path, exc)
            return
        if not isinstance(manifest, dict):
            logger.error("Model manifest has an invalid root object.")
            return
        entries = manifest.get("models", {})
        if not isinstance(entries, dict):
            logger.error("Model manifest has an invalid models section.")
            return
        for component in ("CPU", "GPU"):
            if component not in entries:
                loaded_models.pop(component, None)
                loaded_errors.pop(component, None)
        for component in ("CPU", "GPU"):
            entry = entries.get(component)
            if not isinstance(entry, dict):
                continue
            filename = entry.get("artifact")
            if not isinstance(filename, str) or Path(filename).name != filename:
                loaded_errors[component] = "invalid artifact path"
                continue
            artifact_path = (self.model_dir / filename).resolve()
            if artifact_path.parent != self.model_dir or not artifact_path.is_file():
                loaded_errors[component] = "artifact file is missing"
                continue
            try:
                artifact = joblib.load(artifact_path)
            except Exception as exc:
                loaded_errors[component] = "artifact could not be loaded"
                logger.error("Could not load %s model artifact: %s", component, exc)
                continue
            if (
                not isinstance(artifact, dict)
                or artifact.get("schemaVersion") != 1
                or artifact.get("component") != component
                or artifact.get("featureNames") != FEATURE_NAMES
                or not hasattr(artifact.get("pipeline"), "predict")
            ):
                loaded_errors[component] = "artifact metadata does not match the active feature contract"
                continue
            loaded_models[component] = artifact
            loaded_errors.pop(component, None)
        with self._lock:
            self.models = loaded_models
            self.errors = loaded_errors
            self._manifest_signature = signature

    def model(self, component: str) -> dict[str, Any] | None:
        self.reload_if_changed()
        with self._lock:
            return self.models.get(component)

    def has_models(self) -> bool:
        self.reload_if_changed()
        with self._lock:
            return bool(self.models)

    def health(self) -> dict[str, Any]:
        self.reload_if_changed()
        with self._lock:
            models = dict(self.models)
            errors = dict(self.errors)
        return {
            "status": "UP",
            "modelStatus": "READY" if models else "MODEL_UNAVAILABLE",
            "models": {
                component: {
                    "available": component in models,
                    "model": models.get(component, {}).get("model"),
                    "error": errors.get(component),
                }
                for component in ("CPU", "GPU")
            },
        }


def _to_sensor_point(reading: ReadingRequest) -> SensorPoint:
    return SensorPoint(
        timestamp=reading.timestamp,
        session_id=reading.sessionId or "live",
        cpu_temperature_c=reading.cpuTemperatureC,
        gpu_temperature_c=reading.gpuTemperatureC,
        cpu_utilization_percent=reading.cpuUtilizationPercent,
        cpu_clock_mhz=reading.cpuClockMhz,
        gpu_utilization_percent=reading.gpuUtilizationPercent,
        memory_utilization_percent=reading.memoryUtilizationPercent,
        source_available=reading.sourceAvailable,
    )


def create_app(
    model_dir: Path = MODEL_DIR,
    auto_training_enabled: bool | None = None,
    training_interval_seconds: int | None = None,
    project_root: Path = PROJECT_ROOT,
) -> FastAPI:
    if auto_training_enabled is None:
        auto_setting = os.environ.get("SMARTTHERM_AUTO_TRAINING", "true").strip().lower()
        if auto_setting not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("SMARTTHERM_AUTO_TRAINING must be true or false.")
        auto_training_enabled = auto_setting in {"true", "1", "yes"}
    if training_interval_seconds is None:
        raw_interval = os.environ.get(
            "SMARTTHERM_TRAINING_CHECK_INTERVAL_SECONDS",
            str(DEFAULT_TRAINING_INTERVAL_SECONDS),
        )
        try:
            training_interval_seconds = int(raw_interval)
        except ValueError as exception:
            raise ValueError(
                "SMARTTHERM_TRAINING_CHECK_INTERVAL_SECONDS must be an integer."
            ) from exception
    registry = ModelRegistry(model_dir)
    trainer = BackgroundTrainingCoordinator(
        project_root=project_root,
        model_dir=model_dir,
        interval_seconds=training_interval_seconds,
        enabled=auto_training_enabled,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        trainer.start()
        try:
            yield
        finally:
            trainer.stop()

    app = FastAPI(
        title="SmartTherm local inference",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        result = registry.health()
        result["training"] = trainer.status()
        return result

    @app.post("/training/check")
    def request_training_check() -> dict[str, Any]:
        return trainer.request_check()

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict[str, Any]:
        received_at = datetime.now(timezone.utc)
        readings = request.readings
        newest = readings[-1]
        if not newest.sourceAvailable:
            raise HTTPException(status_code=422, detail="latest sensor reading reports unavailable source")
        age = received_at - newest.timestamp
        if age < timedelta(seconds=-30) or age > MAX_READING_AGE:
            return _unavailable_response(
                readings, received_at, "STALE_DATA", "latest sensor reading is stale"
            )

        points = [_to_sensor_point(reading) for reading in readings]
        continuous = [points[0]]
        for point in points[1:]:
            if (
                point.session_id != continuous[-1].session_id
                or (point.timestamp - continuous[-1].timestamp).total_seconds() > MAX_GAP_SECONDS
            ):
                continuous = [point]
            else:
                continuous.append(point)

        generated_at = received_at
        target_time = generated_at + PREDICTION_HORIZON
        forecasts: list[dict[str, Any]] = []
        for component, temperature_field in (
            ("CPU", "cpuTemperatureC"),
            ("GPU", "gpuTemperatureC"),
        ):
            current = getattr(newest, temperature_field)
            model_artifact = registry.model(component)
            if model_artifact is None:
                forecasts.append(
                    _forecast(
                        component,
                        current,
                        None,
                        generated_at,
                        target_time,
                        "MODEL_UNAVAILABLE",
                        "No validated trained model is available.",
                    )
                )
                continue
            component_readings = [
                point
                for point in continuous
                if getattr(point, f"{component.lower()}_temperature_c") is not None
            ]
            if current is None or len(component_readings) < 3 or (
                component_readings[-1].timestamp - component_readings[0].timestamp
            ).total_seconds() < 10:
                forecasts.append(
                    _forecast(
                        component,
                        current,
                        None,
                        generated_at,
                        target_time,
                        "INSUFFICIENT_HISTORY",
                        "At least three fresh readings spanning ten seconds are required.",
                    )
                )
                continue
            try:
                features = compute_features(continuous, component)
                vector = np.asarray(
                    [[features[name] if features[name] is not None else np.nan
                      for name in FEATURE_NAMES]],
                    dtype=float,
                )
                predicted = float(model_artifact["pipeline"].predict(vector)[0])
                if not math.isfinite(predicted) or not 0 <= predicted <= MAX_TEMPERATURE_C:
                    raise ValueError("model output is outside the supported temperature range")
                forecasts.append(
                    _forecast(
                        component,
                        current,
                        round(predicted, 1),
                        generated_at,
                        target_time,
                        "READY",
                        model_artifact["model"],
                    )
                )
            except (ValueError, TypeError, ArithmeticError, IndexError) as exc:
                logger.error("Prediction failed for %s: %s", component, exc)
                forecasts.append(
                    _forecast(
                        component,
                        current,
                        None,
                        generated_at,
                        target_time,
                        "MODEL_ERROR",
                        "The trained model could not produce a valid estimate.",
                    )
                )

        ready = any(item["status"] == "READY" for item in forecasts)
        available = registry.has_models()
        models = sorted({
            item["model"] for item in forecasts
            if item["status"] == "READY"
        })
        return {
            "status": "READY" if ready else "INSUFFICIENT_HISTORY" if available else "MODEL_UNAVAILABLE",
            "model": "Python trained models: " + ", ".join(models) if models
                else "Python trained models unavailable",
            "trainedModel": ready,
            "generatedAt": generated_at.isoformat(),
            "targetTime": target_time.isoformat(),
            "forecasts": forecasts,
        }

    return app


def _forecast(
    component: str,
    current: float | None,
    predicted: float | None,
    generated_at: datetime,
    target_time: datetime,
    status: str,
    model: str,
) -> dict[str, Any]:
    return {
        "component": component,
        "currentC": current,
        "predictedC": predicted,
        "generatedAt": generated_at.isoformat(),
        "targetTime": target_time.isoformat(),
        "status": status,
        "model": model,
    }


def _unavailable_response(
    readings: list[ReadingRequest],
    generated_at: datetime,
    status: str,
    reason: str,
) -> dict[str, Any]:
    newest = readings[-1]
    target_time = generated_at + PREDICTION_HORIZON
    forecasts = [
        _forecast(
            component,
            getattr(newest, temperature_field),
            None,
            generated_at,
            target_time,
            status,
            reason,
        )
        for component, temperature_field in (
            ("CPU", "cpuTemperatureC"),
            ("GPU", "gpuTemperatureC"),
        )
    ]
    return {
        "status": status,
        "model": "Python trained models unavailable",
        "trainedModel": False,
        "generatedAt": generated_at.isoformat(),
        "targetTime": target_time.isoformat(),
        "forecasts": forecasts,
    }


app = create_app()
