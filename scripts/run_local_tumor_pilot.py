"""Outcome-blind FLARE23 gastric candidate pilot; print aggregates only."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--patient-limit-per-split", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.patient_limit_per_split < 1:
        parser.error("Patient limit must be positive.")
    project = args.project.resolve()
    os.umask(0o077)
    import torch

    from stageworld.artifacts import atomic_write_json, atomic_write_private_json, read_json
    from stageworld.data.gastric_roi import offline_network
    from stageworld.data.paired_ct import HMACPseudonymizer
    from stageworld.data.tumor_roi import TUMOR_ROI_VERSION, TumorSegmenter
    from stageworld.real_workflow import _select_feature_bindings

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    root = project / "artifacts/real/paired-ct-os-v1-first-acquisition/data/restricted"
    assets, splits = (
        read_json(root / name) for name in ("asset_bindings.json", "split_assignments.json")
    )
    approved_root = os.path.commonpath([row["local_path"] for row in assets["bindings"]])
    selected = _select_feature_bindings(
        assets, splits, limit_per_split=args.patient_limit_per_split, include_test=False
    )
    key = os.environ.get("STAGEWORLD_HMAC_KEY_FILE")
    if not key:
        matches = list(Path.home().rglob("gastric-os-v1.hmac"))
        if len(matches) != 1:
            print(json.dumps({"status": "blocked", "error_code": "IDENTITY_BINDING_REQUIRED"}))
            return 2
        key = str(matches[0])
    pseudonymizer = HMACPseudonymizer.from_file(Path(key), project_root=project)
    model_root = Path(
        os.environ.get(
            "STAGEWORLD_TUMOR_MODEL",
            str(Path.home() / ".cache/stageworld-models/flare23-blackbean-20260911"),
        )
    )
    mask_root = Path.home() / ".local/share/stageworld/flare23-gastric-pilot-20260911/tumor_masks"
    output = project / "artifacts/real/flare23-gastric-tumor-pilot-v1"
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    records = []
    counts: Counter[str] = Counter()
    pairs: dict[str, set[str]] = defaultdict(set)
    started = time.monotonic()
    peak = 0
    log_path = output / "restricted_inference.log"
    with log_path.open("a") as log:
        log_path.chmod(0o600)
        saved_out, saved_err = os.dup(1), os.dup(2)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            with (
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
                offline_network(),
            ):
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                segmenter = TumorSegmenter(model_root, mask_root, device=args.device)
                segmenter.initialize()
                for binding, split in selected:
                    try:
                        segmenter.preprocess(
                            binding, approved_root=approved_root, pseudonymizer=pseudonymizer
                        )
                        status = "unreviewed_candidate"
                        pairs[str(binding["patient_id"])].add(str(binding["role"]))
                    except Exception as error:
                        status = str(getattr(error, "code", type(error).__name__))
                        if isinstance(error, torch.OutOfMemoryError):
                            raise
                    counts[status] += 1
                    records.append(
                        {
                            "asset_id": binding["asset_id"],
                            "patient_id": binding["patient_id"],
                            "role": binding["role"],
                            "split": split,
                            "status": status,
                        }
                    )
                    atomic_write_private_json(
                        output / "restricted_manifest.json", {"entries": records}
                    )
                    atomic_write_json(
                        output / "progress.json",
                        {
                            "status": "running",
                            "requested_studies": len(selected),
                            "processed_studies": len(records),
                            "counts": dict(counts),
                        },
                    )
                if args.device == "cuda":
                    peak = torch.cuda.max_memory_allocated()
                result = {
                    "status": "completed",
                    "preprocess_version": TUMOR_ROI_VERSION,
                    "model_artifact_id": segmenter.model_artifact_id,
                    "requested_studies": len(selected),
                    "processed_studies": len(records),
                    "counts": dict(counts),
                    "candidate_complete_pairs": sum(len(x) == 2 for x in pairs.values()),
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_cuda_allocated_bytes": peak,
                    "outcome_data_read": False,
                    "test_features_included": False,
                    "expert_review_completed": False,
                    "clinical_validation": False,
                    "world_model_training_started": False,
                }
                atomic_write_json(output / "pilot_summary.json", result)
                atomic_write_json(output / "progress.json", result)
                status_code = 0
        except Exception as error:
            result = {
                "status": "failed",
                "error_code": str(getattr(error, "code", type(error).__name__)),
                "processed_studies": len(records),
                "counts": dict(counts),
                "outcome_data_read": False,
                "test_features_included": False,
            }
            atomic_write_json(output / "progress.json", result)
            status_code = 1
        finally:
            log.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
    print(json.dumps(result))
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
