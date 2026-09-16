"""Private qingyuan launcher using the existing environment and read-only features."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/project.pcr-recurrence-cv5.yaml")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if project == args.source_project.resolve():
        raise ValueError("Use an isolated destination")
    os.chdir(project)
    os.umask(0o077)
    sys.path.insert(0, str(project / "src"))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(41, 1, 0, 0, 0) != 0:
        raise RuntimeError("Could not disable process huge pages")
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
    for name, value in json.loads(args.bindings.read_text()).items():
        if name.startswith("STAGEWORLD_") and isinstance(value, str):
            os.environ[name] = value
    os.environ["STAGEWORLD_GENERATED_DATA_ROOT"] = str(
        args.source_project / "artifacts/real/paired-ct-os-v1-first-acquisition/data/restricted"
    )
    os.environ["STAGEWORLD_WEIAI_ROOT"] = str(Path(os.environ["STAGEWORLD_CLINICAL_EXCEL"]).parent)
    log_root = project / "artifacts/restricted_logs"
    log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (log_root / "training.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (log_root / ("smoke.log" if args.smoke else "study.log")).open("a") as log:
        saved = os.dup(1), os.dup(2)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                import torch

                from stageworld.binary_workflow import run_binary_study
                from stageworld.config import load_config

                torch.set_num_threads(4)
                torch.set_num_interop_threads(2)
                if not torch.cuda.is_available():
                    raise RuntimeError("This study requires the existing CUDA runtime")
                torch.cuda.set_per_process_memory_fraction(0.1)
                result = run_binary_study(
                    load_config(args.config),
                    smoke=args.smoke,
                    resume=args.resume,
                    audit_only=args.audit_only,
                )
            output = (
                result
                if args.audit_only
                else {
                    "status": result["status"],
                    "study_id": result["study_id"],
                    "fits": len(result["fits"]),
                    "optimizer_updates": result["optimizer_updates"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "test_used": False,
                }
            )
            status = 0
        except BaseException as error:
            traceback.print_exc(file=log)
            output = {
                "status": "failed",
                "error_code": getattr(error, "code", type(error).__name__),
            }
            status = 1
        finally:
            log.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
    print(json.dumps(output))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
