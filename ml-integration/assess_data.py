from __future__ import annotations

import argparse
import json
from pathlib import Path

from smarttherm_ml.dataset import discover_csv_files, load_points
from smarttherm_ml.training import assess_examples


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Assess SmartTherm collector CSVs for real ten-minute training labels."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    args = parser.parse_args()

    files = discover_csv_files(args.project_root)
    points, rejected = load_points(files)
    report = assess_examples(points)
    report["sourceFiles"] = [str(path) for path in files]
    report["rejectedRows"] = dict(rejected)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"CSV files inspected: {len(files)}")
        print(f"Valid sensor rows: {len(points)}")
        print(
            f"CSV session files: {report['sensorSessionCount']}; "
            f"sessions with valid targets: {report['sessionsWithValidTargets']}"
        )
        print("Valid feature readings:", json.dumps(report["featureCoverage"], sort_keys=True))
        print("A target is a real sample 10 minutes later, within 15 seconds, in a continuous session.")
        for component, details in report["components"].items():
            count = details["validTenMinutePairs"]
            split = details["splitCounts"]
            print(
                f"{component}: {count} usable ten-minute examples; "
                f"train={split['train']}, validation={split['validation']}, "
                f"test={split['test']} ({details['trainingStatus']})"
            )
            if details["trainingReason"]:
                print(f"  {details['trainingReason']}")
            if count:
                print(f"  Sessions: {details['sessions']}")
                print(f"  Data span: {details['firstTargetTimestamp']} to {details['lastTargetTimestamp']}")
        if rejected:
            print("Rows rejected:", json.dumps(dict(rejected), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
