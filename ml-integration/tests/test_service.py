from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import joblib
import httpx
from sklearn.dummy import DummyRegressor

from smarttherm_ml.features import FEATURE_NAMES
from smarttherm_ml.service import ModelRegistry, create_app


class ServiceTests(IsolatedAsyncioTestCase):
    async def test_service_lifespan_starts_and_stops_background_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch("smarttherm_ml.service.BackgroundTrainingCoordinator") as coordinator_type:
                app = create_app(
                    Path(temporary),
                    auto_training_enabled=True,
                    project_root=Path(temporary),
                )
                coordinator = coordinator_type.return_value

                async with app.router.lifespan_context(app):
                    coordinator.start.assert_called_once()

                coordinator.stop.assert_called_once()

    async def test_reports_background_training_status_in_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transport = httpx.ASGITransport(
                app=create_app(Path(temporary), auto_training_enabled=False)
            )
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["training"]["enabled"])
        self.assertEqual(response.json()["training"]["status"], "DISABLED")

    async def test_training_check_endpoint_queues_background_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch("smarttherm_ml.service.BackgroundTrainingCoordinator") as coordinator_type:
                coordinator_type.return_value.request_check.return_value = {"status": "QUEUED"}
                transport = httpx.ASGITransport(
                    app=create_app(Path(temporary), auto_training_enabled=True)
                )
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post("/training/check")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "QUEUED")

    def test_registry_reloads_promoted_manifest_and_preserves_model_on_invalid_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            registry = ModelRegistry(model_dir)
            model = DummyRegressor(strategy="constant", constant=63.5)
            model.fit([[0.0] * len(FEATURE_NAMES)], [63.5])
            artifact = {
                "schemaVersion": 2,
                "component": "CPU",
                "model": "XGBoost",
                "targetHorizonSeconds": 300,
                "featureNames": FEATURE_NAMES,
                "pipeline": model,
            }
            joblib.dump(artifact, model_dir / "cpu.joblib")
            manifest_path = model_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({"schemaVersion": 2, "models": {
                    "CPU": {"artifact": "cpu.joblib", "model": "XGBoost"}
                }}),
                encoding="utf-8",
            )

            self.assertTrue(registry.health()["models"]["CPU"]["available"])
            joblib.dump(
                {**artifact, "featureNames": ["invalid"]},
                model_dir / "cpu-invalid-artifact.joblib",
            )
            manifest_path.write_text(
                json.dumps({"schemaVersion": 2, "models": {
                    "CPU": {
                        "artifact": "cpu-invalid-artifact.joblib",
                        "model": "Invalid",
                    }
                }}),
                encoding="utf-8",
            )

            status = registry.health()

        self.assertTrue(status["models"]["CPU"]["available"])
        self.assertEqual(status["models"]["CPU"]["model"], "XGBoost")
        self.assertIn("metadata", status["models"]["CPU"]["error"])

    def test_registry_unloads_removed_model_entries_and_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            model = DummyRegressor(strategy="constant", constant=63.5)
            model.fit([[0.0] * len(FEATURE_NAMES)], [63.5])
            joblib.dump(
                {
                    "schemaVersion": 2,
                    "component": "CPU",
                    "model": "XGBoost",
                    "targetHorizonSeconds": 300,
                    "featureNames": FEATURE_NAMES,
                    "pipeline": model,
                },
                model_dir / "cpu.joblib",
            )
            manifest_path = model_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({"schemaVersion": 2, "models": {
                    "CPU": {"artifact": "cpu.joblib", "model": "XGBoost"}
                }}),
                encoding="utf-8",
            )
            registry = ModelRegistry(model_dir)
            self.assertTrue(registry.health()["models"]["CPU"]["available"])

            manifest_path.write_text(
                json.dumps({"schemaVersion": 2, "models": {}}),
                encoding="utf-8",
            )
            self.assertFalse(registry.health()["models"]["CPU"]["available"])

            manifest_path.write_text(
                json.dumps({"schemaVersion": 2, "models": {
                    "CPU": {"artifact": "cpu.joblib", "model": "XGBoost"}
                }}),
                encoding="utf-8",
            )
            self.assertTrue(registry.health()["models"]["CPU"]["available"])
            manifest_path.unlink()

            status = registry.health()

        self.assertFalse(status["models"]["CPU"]["available"])
        self.assertEqual(status["modelStatus"], "MODEL_UNAVAILABLE")

    async def test_reports_models_unavailable_without_artifact_instead_of_fabricating_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transport = httpx.ASGITransport(app=create_app(Path(temporary)))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/predict", json={"readings": _readings()})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "MODEL_UNAVAILABLE")
        self.assertFalse(response.json()["trainedModel"])
        self.assertIsNone(response.json()["forecasts"][0]["predictedC"])

    async def test_reloads_component_model_and_returns_explicit_trained_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            model = DummyRegressor(strategy="constant", constant=63.5)
            model.fit([[0.0] * len(FEATURE_NAMES)], [63.5])
            artifact = {
                "schemaVersion": 2,
                "component": "CPU",
                "model": "XGBoost",
                "targetHorizonSeconds": 300,
                "featureNames": FEATURE_NAMES,
                "pipeline": model,
            }
            joblib.dump(artifact, model_dir / "cpu.joblib")
            (model_dir / "manifest.json").write_text(
                json.dumps({"schemaVersion": 2, "models": {
                    "CPU": {"artifact": "cpu.joblib", "model": "XGBoost"}
                }}),
                encoding="utf-8",
            )
            transport = httpx.ASGITransport(app=create_app(model_dir))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/predict", json={"readings": _readings()})

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["trainedModel"])
        self.assertEqual(result["forecasts"][0]["status"], "READY")
        self.assertEqual(result["forecasts"][0]["predictedC"], 63.5)
        self.assertEqual(result["forecasts"][1]["status"], "MODEL_UNAVAILABLE")

    async def test_rejects_invalid_reading_and_unsorted_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            transport = httpx.ASGITransport(app=create_app(Path(temporary)))
            invalid_temperature = _readings()
            invalid_temperature[-1]["cpuTemperatureC"] = 200
            unsorted = _readings()
            unsorted[0], unsorted[1] = unsorted[1], unsorted[0]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                invalid_response = await client.post(
                    "/predict", json={"readings": invalid_temperature}
                )
                unsorted_response = await client.post(
                    "/predict", json={"readings": unsorted}
                )

        self.assertEqual(invalid_response.status_code, 422)
        self.assertEqual(unsorted_response.status_code, 422)

    async def test_does_not_join_feature_history_across_collector_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            model = DummyRegressor(strategy="constant", constant=63.5)
            model.fit([[0.0] * len(FEATURE_NAMES)], [63.5])
            joblib.dump(
                {
                    "schemaVersion": 2,
                    "component": "CPU",
                    "model": "XGBoost",
                    "targetHorizonSeconds": 300,
                    "featureNames": FEATURE_NAMES,
                    "pipeline": model,
                },
                model_dir / "cpu.joblib",
            )
            (model_dir / "manifest.json").write_text(
                json.dumps({"schemaVersion": 2, "models": {
                    "CPU": {"artifact": "cpu.joblib", "model": "XGBoost"}
                }}),
                encoding="utf-8",
            )
            readings = _readings()[-2:]
            for reading in readings:
                reading["sessionId"] = "new-collector-session"
            readings[0]["timestamp"] = (
                datetime.now(timezone.utc) - timedelta(seconds=5)
            ).isoformat()
            readings[1]["timestamp"] = datetime.now(timezone.utc).isoformat()
            transport = httpx.ASGITransport(app=create_app(model_dir))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/predict", json={"readings": readings})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["forecasts"][0]["status"], "INSUFFICIENT_HISTORY")
        self.assertIsNone(response.json()["forecasts"][0]["predictedC"])


def _readings() -> list[dict[str, object]]:
    now = datetime.now(timezone.utc)
    readings: list[dict[str, object]] = []
    for index in range(13):
        timestamp = now - timedelta(seconds=(12 - index) * 5)
        readings.append({
            "timestamp": timestamp.isoformat(),
            "sessionId": "test-session",
            "sourceAvailable": True,
            "cpuTemperatureC": 60.0 + index * 0.2,
            "gpuTemperatureC": None,
            "cpuUtilizationPercent": 30.0,
            "cpuClockMhz": 2400.0,
            "gpuUtilizationPercent": None,
            "memoryUtilizationPercent": 45.0,
        })
    return readings
