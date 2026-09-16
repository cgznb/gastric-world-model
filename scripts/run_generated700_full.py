"""Run training, independent metric audit and historical paired comparison serially."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--reference-project", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    stages = [
        (
            "training_and_verification",
            [str(args.project / "scripts/run_generated700.py"), *forwarded],
        ),
        ("independent_metric_audit", [str(args.project / "audit_results.py"), str(args.project)]),
        (
            "historical_paired_comparison",
            [
                str(args.project / "scripts/compare_generated700.py"),
                "--new",
                str(args.project / "artifacts/formal"),
                "--old",
                str(args.reference_project / "artifacts/formal"),
            ],
        ),
    ]
    status = args.project / "artifacts/controller.json"
    for stage, command in stages:
        status.write_text(
            json.dumps(
                {"stage": stage, "status": "running", "time_utc": datetime.now(UTC).isoformat()}
            )
        )
        capture = stage == "independent_metric_audit"
        result = subprocess.run(
            [sys.executable, *command], check=False, capture_output=capture, text=True
        )
        if capture:
            print(result.stdout, flush=True)
            if result.returncode:
                print(result.stderr, file=sys.stderr, flush=True)
            else:
                report = json.loads(result.stdout)
                (args.project / "artifacts/formal/metric_verification.json").write_text(
                    json.dumps(report, indent=2)
                )
        if result.returncode:
            status.write_text(
                json.dumps({"stage": stage, "status": "failed", "exit_code": result.returncode})
            )
            raise SystemExit(result.returncode)
    status.write_text(
        json.dumps({"status": "completed", "time_utc": datetime.now(UTC).isoformat()})
    )


if __name__ == "__main__":
    main()
