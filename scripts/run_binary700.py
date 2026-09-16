"""Run the authorized original700 nested binary-endpoint study."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"

import torch  # noqa: E402

from stageworld.artifacts import read_json  # noqa: E402
from stageworld.binary700_data import prepare_pool  # noqa: E402
from stageworld.binary700_verification import verify_recovery, verify_study  # noqa: E402
from stageworld.binary700_workflow import run_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--former-test-features", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path)
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
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.1)
    pool = prepare_pool(args.source, args.former_test_features, args.bindings, args.pool)
    audit = read_json(args.pool / "audit.json")
    print(
        {
            k: audit[k]
            for k in (
                "patients",
                "label_counts",
                "ct0_available",
                "ct1_available",
                "complete_ct_pairs",
                "patients_excluded",
            )
        },
        flush=True,
    )
    if args.prepare_only:
        return
    if args.output is None:
        parser.error("--output is required for training")
    if args.verify_only:
        print(verify_study(pool, args.output), flush=True)
        return
    print(
        run_study(
            pool,
            read_json(args.pool / "folds.json"),
            args.output,
            smoke=args.smoke,
            replicates=25 if args.smoke else 1000,
        ),
        flush=True,
    )
    if args.smoke:
        print(verify_recovery(pool, args.output), flush=True)
    print(verify_study(pool, args.output), flush=True)


if __name__ == "__main__":
    main()
