"""Run the authorized complete651 fivefold study and publish per-seed reports."""

import argparse
import fcntl
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"

import torch  # noqa: E402

from stageworld.artifacts import atomic_write_private_json, read_json  # noqa: E402
from stageworld.generated651_data import prepare_pool  # noqa: E402
from stageworld.generated651_verification import verify_recovery, verify_study  # noqa: E402
from stageworld.generated651_workflow import run_study  # noqa: E402


def bind_runtime(project: Path, output: Path) -> dict:
    paths = sorted((project / "src").rglob("*.py")) + [Path(__file__).resolve()]
    actual = {
        str(path.relative_to(project)): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    }
    contract = {"files": actual, "identity_policy": "immutable_source_metadata_no_checksums"}
    target = output / "runtime_source.json"
    if target.exists() and read_json(target) != contract:
        raise ValueError("Runtime source changed since this study started")
    atomic_write_private_json(target, contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.1)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.output.parent / "generated651.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool = prepare_pool(args.source_pool, args.pool)
        audit = read_json(args.pool / "audit.json")
        print(
            {
                key: audit[key]
                for key in (
                    "patients",
                    "complete_CT_pairs",
                    "pcr",
                    "recurrence",
                    "excluded_patients",
                )
            },
            flush=True,
        )
        if args.prepare_only:
            return
        project = Path(__file__).resolve().parents[1]
        original = bind_runtime(project, args.output)
        controller = args.output.parent / f"{args.output.name}_controller.json"
        atomic_write_private_json(
            controller, {"status": "running", "time_utc": datetime.now(UTC).isoformat()}
        )
        try:
            if args.verify_only:
                print(verify_study(pool, args.output), flush=True)
            else:
                result = run_study(
                    pool, read_json(args.pool / "folds.json"), args.output, smoke=args.smoke
                )
                print(result, flush=True)
                if args.smoke:
                    print(verify_recovery(pool, args.output), flush=True)
                verification = verify_study(pool, args.output)
                print(verification, flush=True)
                if bind_runtime(project, args.output) != original:
                    raise ValueError("Runtime changed during training")
                atomic_write_private_json(
                    args.output / "completed.json",
                    {
                        **result,
                        "status": "completed",
                        "verification": verification,
                        "time_utc": datetime.now(UTC).isoformat(),
                    },
                )
        except BaseException as error:
            atomic_write_private_json(
                controller,
                {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "time_utc": datetime.now(UTC).isoformat(),
                },
            )
            raise
        atomic_write_private_json(
            controller, {"status": "completed", "time_utc": datetime.now(UTC).isoformat()}
        )


if __name__ == "__main__":
    main()
