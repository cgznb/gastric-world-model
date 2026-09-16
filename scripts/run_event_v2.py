"""Run a locked ten-seed 80/10/10 internal-holdout event study."""

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
from stageworld.event_data import fit_inputs, prepare_pool  # noqa: E402
from stageworld.event_v2_spec import FAMILIES, specification  # noqa: E402
from stageworld.event_v2_training import train_phase  # noqa: E402
from stageworld.event_v2_verification import verify_study  # noqa: E402
from stageworld.event_v2_workflow import prepare_splits, run_study, subset_pool  # noqa: E402


def bind_runtime(project: Path, output: Path) -> None:
    files = sorted((project / "src").rglob("*.py")) + [Path(__file__).resolve()]
    contract = {
        "identity_policy": "immutable_source_metadata_no_checksums",
        "files": {
            str(path.relative_to(project)): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        },
    }
    path = output / "runtime_source.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("Runtime source changed after this experiment started")
    if not path.exists():
        atomic_write_private_json(path, contract)


def smoke(pool, output: Path) -> dict:
    from stageworld.event_v2_splits import make_split

    split = make_split(pool, specification()["seeds"][0])
    groups = split["patient_ids"]
    ids = groups["train"][:32] + groups["validation"][:32]
    dev = subset_pool(pool, ids)
    x, snapshot = fit_inputs(dev, ids[:32], output / "inputs.pt")
    train, validation = torch.arange(32), torch.arange(32, 64)
    reports = []
    for family in ("event_v1", "event_v2"):
        target = output / family
        for phase in ("pretrain", "joint"):
            model, report = train_phase(
                dev,
                x,
                snapshot,
                train,
                validation,
                target / phase,
                family=family,
                phase=phase,
                seed=17,
                epochs=2,
                parent=None if phase == "pretrain" else target / "pretrain/selected.pt",
            )
            if report["device"] != "cuda":
                raise RuntimeError("Real-data smoke must use CUDA")
            reports.append(report)
            del model
    result = {
        "status": "passed",
        "training_patients": 32,
        "validation_patients": 32,
        "test_patients_evaluated": 0,
        "test_patients_used_for_fitting": 0,
        "phases": len(reports),
        "optimizer_updates": sum(report["updates"] for report in reports),
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "architecture_or_hyperparameter_selection_from_smoke": False,
    }
    atomic_write_private_json(output / "smoke.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=specification()["families"]
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
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
    with (args.output.parent / "event_v2.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool = prepare_pool(args.source_pool, args.bindings, args.pool)
        spec = {**specification(), "families": args.families}
        if args.prepare_only:
            prepare_splits(pool, args.output, spec)
            print(read_json(args.output / "split_audit.json"), flush=True)
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This entrypoint requires a BF16 CUDA GPU")
        torch.cuda.set_per_process_memory_fraction(0.2)
        project = Path(__file__).resolve().parents[1]
        bind_runtime(project, args.output)
        controller = args.output.parent / f"{args.output.name}_controller.json"

        def status(value: str, **extra) -> None:
            atomic_write_private_json(
                controller,
                {
                    "status": value,
                    "time_utc": datetime.now(UTC).isoformat(),
                    "pid": os.getpid(),
                    **extra,
                },
            )

        status("running", device=torch.cuda.get_device_name(0))
        try:
            if args.smoke:
                result = smoke(pool, args.output)
            elif args.verify_only:
                result = verify_study(pool, args.output)
            else:
                run_study(pool, args.output, spec=spec)
                verification = verify_study(pool, args.output)
                result = {
                    "status": "completed",
                    "seeds": spec["seeds"],
                    "families": spec["families"],
                    "verification": verification,
                    "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                }
                atomic_write_private_json(args.output / "completed.json", result)
                atomic_write_private_json(args.output / "progress.json", result)
            bind_runtime(project, args.output)
            print(result, flush=True)
        except BaseException as error:
            status("failed", exception_type=type(error).__name__)
            raise
        status("completed")


if __name__ == "__main__":
    main()
