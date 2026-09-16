"""Run the isolated generated-S1 study using explicit private host bindings."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import os
import traceback
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/project.generated-s1-baseline19.yaml")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if project == args.source_project.resolve():
        raise ValueError("An isolated destination project is required.")
    os.chdir(project)
    os.umask(0o077)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise OSError(ctypes.get_errno(), "Could not disable process huge pages")
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
    bindings = json.loads(args.bindings.read_text())
    for name, value in bindings.items():
        if name.startswith("STAGEWORLD_") and isinstance(value, str):
            os.environ[name] = value
    os.environ["STAGEWORLD_GENERATED_DATA_ROOT"] = str(
        args.source_project / "artifacts/real/paired-ct-os-v1-first-acquisition/data/restricted"
    )
    os.environ["STAGEWORLD_WEIAI_ROOT"] = str(Path(os.environ["STAGEWORLD_CLINICAL_EXCEL"]).parent)
    import torch

    from stageworld.config import load_config
    from stageworld.generated_workflow import run_generated_study

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    log_root = project / "artifacts/restricted_logs"
    log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (log_root / "training.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_path = log_root / ("smoke.log" if args.smoke else "study.log")
    with log_path.open("a") as log:
        saved = os.dup(1), os.dup(2)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                from finalize_generated_report import finalize
                from verify_generated_inference import verify

                config = load_config(args.config)
                result: dict[str, Any]
                if config.training.development_protocol == "ct6-s1-pred-only-v1":
                    from stageworld.ct6_workflow import prepare_ct6_bundle
                    from stageworld.s1_only_workflow import run_s1_only_study, validate_protocol

                    validate_protocol(config)
                    if not torch.cuda.is_available():
                        raise RuntimeError("This real study requires the approved CUDA runtime.")
                    torch.cuda.set_per_process_memory_fraction(0.1)
                    if args.audit_only:
                        bundle, _ = prepare_ct6_bundle(config)
                        result = {
                            "status": "audited_not_trained",
                            "train": bundle.training_patient_count,
                            "validation": bundle.validation_patient_count,
                            "test_used": False,
                        }
                    else:
                        result = run_s1_only_study(config, smoke=args.smoke, resume=args.resume)
                elif (config.training.development_protocol or "").startswith("ct6-"):
                    from stageworld.ct6_workflow import prepare_ct6_bundle, run_ct6_study

                    if args.audit_only:
                        bundle, _ = prepare_ct6_bundle(config)
                        result = {
                            "status": "audited_not_trained",
                            "train": bundle.training_patient_count,
                            "validation": bundle.validation_patient_count,
                            "test_used": False,
                        }
                    else:
                        result = run_ct6_study(config, smoke=args.smoke, resume=args.resume)
                else:
                    if args.resume is not None or args.audit_only:
                        raise ValueError("These options require the CT6 protocol.")
                    result = run_generated_study(config, smoke=args.smoke)
                    verify(project / config.output_root / result["study_id"])
                    result["independent_inference_verified"] = True
                    finalize(project / config.output_root / result["study_id"])
                    result["final_artifact_verification"] = "passed"
            status = 0
        except BaseException as error:
            traceback.print_exc(file=log)
            result = {
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
    print(json.dumps(result))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
