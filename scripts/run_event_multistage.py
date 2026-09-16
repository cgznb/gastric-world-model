"""Launch full event training. Real-data small-scale trials are intentionally absent."""

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
from stageworld.event_data import prepare_pool  # noqa: E402
from stageworld.event_verification import verify_study  # noqa: E402
from stageworld.event_workflow import run_study, validate_folds  # noqa: E402


def bind_runtime(project: Path, output: Path) -> dict:
    paths = sorted((project / "src").rglob("*.py")) + [Path(__file__).resolve()]
    contract = {
        "files": {
            str(path.relative_to(project)): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        },
        "identity_policy": "immutable_source_metadata_no_checksums",
    }
    target = output / "runtime_source.json"
    if target.exists() and read_json(target) != contract:
        raise ValueError("Runtime source changed after this study started")
    atomic_write_private_json(target, contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.output.parent / "event_multistage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool = prepare_pool(args.source_pool, args.bindings, args.pool)
        folds = read_json(args.source_pool / "folds.json")
        validate_folds(pool, folds)
        print(read_json(args.pool / "audit.json"), flush=True)
        if args.prepare_only:
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Formal training requires the assigned BF16 CUDA GPU")
        torch.cuda.set_per_process_memory_fraction(0.1)
        project = Path(__file__).resolve().parents[1]
        bind_runtime(project, args.output)
        controller = args.output.parent / f"{args.output.name}_controller.json"

        def status(name: str, **extra) -> None:
            atomic_write_private_json(
                controller,
                {
                    "status": name,
                    "time_utc": datetime.now(UTC).isoformat(),
                    "pid": os.getpid(),
                    **extra,
                },
            )

        status("running", device=torch.cuda.get_device_name(0))
        try:
            if args.verify_only:
                print(verify_study(pool, folds, args.output), flush=True)
            else:
                result = run_study(pool, folds, args.output)
                verification = verify_study(pool, folds, args.output)
                bind_runtime(project, args.output)
                result = {**result, "status": "completed", "verification": verification}
                atomic_write_private_json(args.output / "completed.json", result)
                atomic_write_private_json(args.output / "progress.json", result)
                print(result, flush=True)
        except BaseException as error:
            status("failed", exception_type=type(error).__name__)
            raise
        status("completed")


if __name__ == "__main__":
    main()
