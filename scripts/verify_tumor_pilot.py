"""Verify saved pilot mask geometry, class identity and split isolation; no images emitted."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from stageworld.artifacts import atomic_write_json, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve() / "artifacts/real/flare23-gastric-tumor-pilot-v1"
    summary = read_json(root / "pilot_summary.json")
    assert summary["status"] == "completed"
    rows = read_json(root / "restricted_manifest.json")["entries"]
    masks = Path.home() / ".local/share/stageworld/flare23-gastric-pilot-20260911/tumor_masks"
    counts: Counter[str] = Counter()
    for row in rows:
        assert row["split"] in ("train", "validation")
        asset = row["asset_id"]
        original = read_json(masks / f"{asset}.json")
        full = nib.load(masks / f"{asset}.nii.gz")
        candidate = nib.load(masks / f"{asset}.candidate.nii.gz")
        labels = np.asarray(full.dataobj)
        binary = np.asarray(candidate.dataobj)
        assert list(full.shape) == original["source_shape"] == list(candidate.shape)
        assert np.allclose(full.affine, original["source_affine"], rtol=0, atol=1e-4)
        assert np.allclose(candidate.affine, full.affine, rtol=0, atol=1e-4)
        assert np.isin(labels, range(15)).all() and np.isin(binary, (0, 1)).all()
        assert np.all(labels[binary > 0] == 14)
        assert original["model_artifact_id"] == summary["model_artifact_id"]
        counts["geometry_and_label_checks_passed"] += 1
        counts["with_any_pan_cancer_prediction"] += int((labels == 14).any())
        counts["with_gastric_candidate"] += int(binary.any())
        counts["with_stomach_prediction"] += int((labels == 11).any())
        if (labels == 2).any() and (labels == 13).any():
            left, right = ndimage.center_of_mass(np.ones(labels.shape, np.uint8), labels, (13, 2))
            left_ras = full.affine @ np.array([*left, 1])
            right_ras = full.affine @ np.array([*right, 1])
            counts["bilateral_kidney_order_checked"] += 1
            counts["expected_kidney_left_right_order"] += int(left_ras[0] < right_ras[0])
    unsafe = sum((path.stat().st_mode & 0o077) != 0 for path in masks.rglob("*"))
    assert unsafe == 0
    assert len(rows) == summary["requested_studies"] == summary["processed_studies"]
    result = {
        "status": "ok", "studies_checked": len(rows), "counts": dict(counts),
        "unsafe_permissions": unsafe, "test_entries": 0, "outcome_data_read": False,
        "expert_review_completed": False, "segmentation_accuracy_measured": False,
        "limitation": "Geometry and anatomy-order checks do not validate gastric tumor identity.",
    }
    atomic_write_json(root / "pilot_verification.json", result)
    import json

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
