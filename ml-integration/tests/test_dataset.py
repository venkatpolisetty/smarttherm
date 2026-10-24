from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from smarttherm_ml.dataset import SensorPoint, TrainingExample, build_examples, load_points
from smarttherm_ml.features import FEATURE_NAMES, compute_features
from smarttherm_ml.training import (
    InsufficientTrainingDataError,
    assess_examples,
    split_chronologically,
    train_component,
)
from smarttherm_ml.dataset import TrainingExample


class DatasetTests(unittest.TestCase):
    def test_legacy_csv_sensor_names_are_normalized_without_inventing_cpu_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry_legacy.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Timestamp", "CPU_Load", "GPU_Temp"],
                )
                writer.writeheader()
                writer.writerow(
                    {"Timestamp": "2026-10-20 12:00:00", "CPU_Load": "37", "GPU_Temp": "43"}
                )

            points, rejected = load_points([path])

        self.assertEqual(len(points), 1)
        self.assertIsNone(points[0].cpu_temperature_c)
        self.assertEqual(points[0].gpu_temperature_c, 43)
        self.assertEqual(points[0].cpu_utilization_percent, 37)
        self.assertEqual(rejected["duplicate_timestamp"], 0)

    def test_does_not_create_targets_using_a_different_session(self) -> None:
        start = datetime(2026, 10, 20, tzinfo=timezone.utc)
        points = [
            _point(start + timedelta(seconds=5 * index), "first")
            for index in range(10)
        ]
        points.extend(
            _point(start + timedelta(seconds=600 + 5 * index), "second")
            for index in range(10)
        )

        examples, _ = build_examples(points)

        self.assertEqual(examples["CPU"], [])
        self.assertEqual(examples["GPU"], [])

    def test_builds_real_ten_minute_targets_inside_continuous_session(self) -> None:
        start = datetime(2026, 10, 20, tzinfo=timezone.utc)
        points = [
            _point(start + timedelta(seconds=5 * index), "continuous")
            for index in range(240)
        ]

        examples, _ = build_examples(points)

        self.assertGreater(len(examples["CPU"]), 0)
        first = examples["CPU"][0]
        target_index = round((first.timestamp - start).total_seconds() / 5) + 120
        self.assertEqual(
            points[target_index].timestamp - first.timestamp,
            timedelta(seconds=600),
        )
        self.assertEqual(first.target_c, points[target_index].cpu_temperature_c)
        self.assertEqual(set(first.features), set(FEATURE_NAMES))
        self.assertEqual(first.persistence_prediction_c, first.features["temperature_current_c"])

    def test_feature_history_uses_only_rows_at_or_before_the_prediction_time(self) -> None:
        start = datetime(2026, 10, 20, tzinfo=timezone.utc)
        history = [_point(start + timedelta(seconds=5 * i), "one") for i in range(5)]
        features = compute_features(history, "CPU")

        self.assertEqual(features["temperature_current_c"], history[-1].cpu_temperature_c)
        self.assertAlmostEqual(features["temperature_slope_30s_c_per_min"], 1.0)

    def test_chronological_split_refuses_short_data(self) -> None:
        start = datetime(2026, 10, 20, tzinfo=timezone.utc)
        examples, _ = build_examples(
            [_point(start + timedelta(seconds=5 * index), "one") for index in range(240)]
        )

        with self.assertRaises(InsufficientTrainingDataError):
            split_chronologically(examples["CPU"])

    def test_chronological_split_keeps_whole_time_blocks_and_embargoes_targets(self) -> None:
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        examples = [
            TrainingExample(
                timestamp=start + timedelta(hours=hour, minutes=minute),
                session_id="long-running-collector",
                component="CPU",
                features={name: 1.0 for name in FEATURE_NAMES},
                target_c=50.0,
                persistence_prediction_c=49.0,
            )
            for hour in range(50)
            for minute in range(60)
        ]

        train, validation, test = split_chronologically(
            examples, minimum_train=100, minimum_evaluation=50
        )

        self.assertTrue(train[-1].timestamp + timedelta(minutes=10) < validation[0].timestamp)
        self.assertTrue(validation[-1].timestamp + timedelta(minutes=10) < test[0].timestamp)
        block = lambda item: int((item.timestamp - start).total_seconds() // 3600)
        train_blocks = {block(item) for item in train}
        validation_blocks = {block(item) for item in validation}
        test_blocks = {block(item) for item in test}
        self.assertTrue(train_blocks.isdisjoint(validation_blocks))
        self.assertTrue(validation_blocks.isdisjoint(test_blocks))
        self.assertLess(max(train_blocks), min(validation_blocks))
        self.assertLess(max(validation_blocks), min(test_blocks))

    def test_assessment_reports_actual_splits_and_data_gate_status(self) -> None:
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        points = [
            _point(start + timedelta(seconds=5 * index), "assessment-session")
            for index in range(4 * 60 * 60 // 5)
        ]

        report = assess_examples(points)

        for component in ("CPU", "GPU"):
            details = report["components"][component]
            self.assertEqual(
                set(details["splitCounts"]),
                {"train", "validation", "test"},
            )
            self.assertGreater(details["splitCounts"]["train"], 0)
            self.assertGreater(details["splitCounts"]["validation"], 0)
            self.assertGreater(details["splitCounts"]["test"], 0)
            self.assertEqual(details["trainingStatus"], "SUFFICIENT_DATA")
            self.assertIsNone(details["trainingReason"])

    def test_trains_competing_models_and_persists_an_evaluated_candidate(self) -> None:
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        examples: list[TrainingExample] = []
        for hour in range(40):
            for minute in range(60):
                current = 35.0 + (minute % 20)
                features = {name: current for name in FEATURE_NAMES}
                examples.append(
                    TrainingExample(
                        timestamp=start + timedelta(hours=hour, minutes=minute),
                        session_id="synthetic-test-session",
                        component="CPU",
                        features=features,
                        target_c=30.0 + current * 0.8,
                        persistence_prediction_c=current,
                    )
                )
        with tempfile.TemporaryDirectory() as temporary:
            report = train_component(
                "CPU", examples, Path(temporary)
            )
            self.assertEqual(report["status"], "PROMOTED", report)
            artifact = Path(report["artifact"])
            self.assertTrue(artifact.is_file())
            self.assertIn("cpu-", artifact.name)
            self.assertTrue((Path(temporary) / "manifest.json").is_file())
            self.assertLess(report["test"]["mae_c"], report["testBaseline"]["mae_c"])

    def test_does_not_promote_candidate_when_persistence_baseline_is_perfect(self) -> None:
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        examples = [
            TrainingExample(
                timestamp=start + timedelta(hours=hour, minutes=minute),
                session_id="flat-temperature-session",
                component="CPU",
                features={name: 50.0 for name in FEATURE_NAMES},
                target_c=50.0,
                persistence_prediction_c=50.0,
            )
            for hour in range(40)
            for minute in range(60)
        ]

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            _install_incumbent(model_dir, "CPU", LinearRegression().fit([[50.0]], [50.0]))
            previous_manifest = (model_dir / "manifest.json").read_text(encoding="utf-8")

            report = train_component("CPU", examples, model_dir)

            self.assertEqual(report["status"], "NOT_PROMOTED")
            self.assertEqual((model_dir / "manifest.json").read_text(encoding="utf-8"), previous_manifest)

    def test_does_not_replace_a_better_active_model(self) -> None:
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        examples = []
        for hour in range(40):
            for minute in range(60):
                current = 35.0 + (minute % 20)
                features = {name: current for name in FEATURE_NAMES}
                examples.append(
                    TrainingExample(
                        timestamp=start + timedelta(hours=hour, minutes=minute),
                        session_id="incumbent-comparison-session",
                        component="CPU",
                        features=features,
                        target_c=20.0 + current * 0.5,
                        persistence_prediction_c=current,
                    )
                )
        incumbent_model = LinearRegression().fit(
            [
                [item.features[name] for name in FEATURE_NAMES]
                for item in examples
            ],
            [item.target_c for item in examples],
        )

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            _install_incumbent(model_dir, "CPU", incumbent_model)
            previous_manifest = (model_dir / "manifest.json").read_text(encoding="utf-8")

            report = train_component("CPU", examples, model_dir)

            self.assertEqual(report["status"], "NOT_PROMOTED")
            self.assertIn("regress the active model", report["reason"])
            self.assertEqual((model_dir / "manifest.json").read_text(encoding="utf-8"), previous_manifest)


def _point(timestamp: datetime, session_id: str) -> SensorPoint:
    elapsed_minutes = (timestamp.minute + timestamp.second / 60.0)
    temperature = 40.0 + elapsed_minutes
    return SensorPoint(
        timestamp=timestamp,
        session_id=session_id,
        cpu_temperature_c=temperature,
        gpu_temperature_c=temperature + 2.0,
        cpu_utilization_percent=25.0,
        cpu_clock_mhz=2400.0,
        gpu_utilization_percent=10.0,
        memory_utilization_percent=45.0,
    )


def _install_incumbent(model_dir: Path, component: str, pipeline: object) -> None:
    artifact_name = f"{component.lower()}-incumbent.joblib"
    joblib.dump(
        {
            "schemaVersion": 1,
            "component": component,
            "model": "Incumbent",
            "featureNames": FEATURE_NAMES,
            "pipeline": pipeline,
        },
        model_dir / artifact_name,
    )
    (model_dir / "manifest.json").write_text(
        json.dumps({"schemaVersion": 1, "models": {
            component: {"artifact": artifact_name, "model": "Incumbent"}
        }}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
