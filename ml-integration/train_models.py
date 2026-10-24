from __future__ import annotations

import argparse
import json
from pathlib import Path

from smarttherm_ml.training import (
    MIN_EVALUATION_EXAMPLES,
    MIN_TRAINING_EXAMPLES,
    train_from_history,
)


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train and chronologically evaluate SmartTherm ten-minute regressors."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument("--min-train", type=int, default=MIN_TRAINING_EXAMPLES)
    parser.add_argument("--min-evaluation", type=int, default=MIN_EVALUATION_EXAMPLES)
    args = parser.parse_args()
    if args.min_train < 1 or args.min_evaluation < 1:
        parser.error("--min-train and --min-evaluation must be positive")

    report = train_from_history(
        args.project_root,
        args.model_dir,
        args.min_train,
        args.min_evaluation,
    )
    print(f"CSV files inspected: {report['csvFilesInspected']}")
    print(f"Valid sensor rows: {report['validSensorRows']}")
    print("Current data is not replaced with synthetic examples.")
    print("Rejected sensor rows:", json.dumps(report["rejectedSensorRows"], sort_keys=True))
    print("Rejected target candidates:", json.dumps(report["rejectedTargetCandidates"], sort_keys=True))
    for component in ("CPU", "GPU"):
        print(json.dumps(report["components"][component], indent=2))
    return 0 if report["status"] == "PROMOTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
