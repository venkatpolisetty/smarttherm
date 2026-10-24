from __future__ import annotations

import csv
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from smarttherm_ml.auto_training import BackgroundTrainingCoordinator
from smarttherm_ml.training import _training_process_lock


class BackgroundTrainingTests(unittest.TestCase):
    def test_reports_insufficient_data_without_creating_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_short_training_file(root)
            model_dir = root / "models"
            coordinator = BackgroundTrainingCoordinator(root, model_dir, 3600)

            status = coordinator.check_once()

            self.assertEqual(status["status"], "INSUFFICIENT_DATA")
            self.assertEqual(status["components"]["CPU"]["status"], "INSUFFICIENT_DATA")
            self.assertFalse((model_dir / "manifest.json").exists())
            self.assertTrue((model_dir / "auto-training-state.json").is_file())

    def test_skips_unchanged_data_and_retries_after_history_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_file = root / "telemetry.csv"
            data_file.write_text("Timestamp,CPU_Temp\n", encoding="utf-8")
            coordinator = BackgroundTrainingCoordinator(root, root / "models", 3600)
            result = {
                "status": "INSUFFICIENT_DATA",
                "validSensorRows": 0,
                "csvFilesInspected": 1,
                "components": {
                    "CPU": {"component": "CPU", "status": "INSUFFICIENT_DATA"},
                    "GPU": {"component": "GPU", "status": "INSUFFICIENT_DATA"},
                },
            }
            with patch("smarttherm_ml.auto_training.train_from_history", return_value=result) as train:
                coordinator.check_once()
                coordinator.check_once()
                self.assertEqual(train.call_count, 1)
                data_file.write_text("Timestamp,CPU_Temp\nnew row\n", encoding="utf-8")
                coordinator.check_once()

            self.assertEqual(train.call_count, 2)

    def test_request_check_queues_a_background_training_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "telemetry.csv").write_text("Timestamp,CPU_Temp\n", encoding="utf-8")
            coordinator = BackgroundTrainingCoordinator(root, root / "models", 3600)
            entered_training = threading.Event()
            release_training = threading.Event()
            finished_check = threading.Event()
            result = {
                "status": "INSUFFICIENT_DATA",
                "validSensorRows": 0,
                "csvFilesInspected": 1,
                "components": {
                    "CPU": {"component": "CPU", "status": "INSUFFICIENT_DATA"},
                    "GPU": {"component": "GPU", "status": "INSUFFICIENT_DATA"},
                },
            }

            def blocked_training(*_: object) -> dict[str, object]:
                entered_training.set()
                if not release_training.wait(timeout=5):
                    raise TimeoutError("test did not release training worker")
                return result

            original_check = coordinator.check_once

            def tracked_check() -> dict[str, object]:
                try:
                    return original_check()
                finally:
                    finished_check.set()

            with patch("smarttherm_ml.auto_training.train_from_history", side_effect=blocked_training), \
                    patch.object(coordinator, "check_once", side_effect=tracked_check):
                queued = coordinator.request_check()
                self.assertIn(queued["status"], {"QUEUED", "CHECKING"})
                self.assertTrue(entered_training.wait(timeout=5))
                release_training.set()
                self.assertTrue(finished_check.wait(timeout=5))

    def test_surfaces_training_failures_without_terminating_the_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "telemetry.csv").write_text("Timestamp,CPU_Temp\n", encoding="utf-8")
            coordinator = BackgroundTrainingCoordinator(root, root / "models", 3600)
            with patch(
                "smarttherm_ml.auto_training.train_from_history",
                side_effect=RuntimeError("training backend failed"),
            ):
                status = coordinator.check_once()

            self.assertEqual(status["status"], "TRAINING_ERROR")
            self.assertIn("training backend failed", status["error"])

    def test_prevents_overlapping_training_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "telemetry.csv").write_text("Timestamp,CPU_Temp\n", encoding="utf-8")
            coordinator = BackgroundTrainingCoordinator(root, root / "models", 3600)
            entered_training = threading.Event()
            release_training = threading.Event()
            result = {
                "status": "INSUFFICIENT_DATA",
                "validSensorRows": 0,
                "csvFilesInspected": 1,
                "components": {
                    "CPU": {"component": "CPU", "status": "INSUFFICIENT_DATA"},
                    "GPU": {"component": "GPU", "status": "INSUFFICIENT_DATA"},
                },
            }

            def blocked_training(*_: object) -> dict[str, object]:
                entered_training.set()
                if not release_training.wait(timeout=5):
                    raise TimeoutError("test did not release training worker")
                return result

            with patch("smarttherm_ml.auto_training.train_from_history", side_effect=blocked_training) as train:
                worker = threading.Thread(target=coordinator.check_once)
                worker.start()
                self.assertTrue(entered_training.wait(timeout=5))
                status = coordinator.check_once()
                release_training.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(train.call_count, 1)
            self.assertEqual(status["status"], "CHECKING")

    def test_prevents_concurrent_training_processes_using_the_same_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "models"
            with _training_process_lock(model_dir) as first:
                with _training_process_lock(model_dir) as second:
                    self.assertTrue(first)
                    self.assertFalse(second)

    def test_rejects_unreasonably_short_training_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "at least 60 seconds"):
                BackgroundTrainingCoordinator(Path(temporary), Path(temporary) / "models", 1)


def _write_short_training_file(root: Path) -> None:
    path = root / "training_telemetry_test.csv"
    fields = [
        "schemaVersion",
        "sessionId",
        "timestamp",
        "cpuTemperatureC",
        "gpuTemperatureC",
        "cpuUtilizationPercent",
        "cpuClockMhz",
        "cpuPowerW",
        "cpuFanSpeedRpm",
        "gpuUtilizationPercent",
        "memoryUtilizationPercent",
        "gpuSensor",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "schemaVersion": 1,
            "sessionId": "short-session",
            "timestamp": "2026-10-20T12:00:00+00:00",
            "cpuTemperatureC": 60.0,
            "cpuUtilizationPercent": 25.0,
            "cpuClockMhz": 2400.0,
            "memoryUtilizationPercent": 40.0,
        })


if __name__ == "__main__":
    unittest.main()
