"""Outcome-blind distance sensitivity; this does not change the locked candidate rule."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from stageworld.artifacts import atomic_write_json, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve() / "artifacts/real/flare23-gastric-tumor-pilot-v1"
    rows = read_json(root / "restricted_manifest.json")["entries"]
    masks = Path.home() / ".local/share/stageworld/flare23-gastric-pilot-20260911/tumor_masks"
    thresholds = (10, 20, 40, 80)
    positive: dict[int, dict[str, set[str]]] = {t: defaultdict(set) for t in thresholds}
    distances = []
    for row in rows:
        image = nib.load(masks / f"{row['asset_id']}.nii.gz")
        labels = np.asarray(image.dataobj)
        surfaces = []
        for value in (11, 14):
            mask = labels == value
            boundary = mask & ~ndimage.binary_erosion(mask)
            positions = np.argwhere(boundary)
            surfaces.append(positions @ image.affine[:3, :3].T + image.affine[:3, 3])
        if any(len(surface) == 0 for surface in surfaces):
            continue
        distance = float(cKDTree(surfaces[0]).query(surfaces[1], workers=1)[0].min())
        distances.append(distance)
        for threshold in thresholds:
            if distance <= threshold:
                positive[threshold][row["patient_id"]].add(row["role"])
    result = {
        "status": "completed", "studies": len(rows),
        "with_both_stomach_and_pan_cancer_predictions": len(distances),
        "minimum_distance_mm_range": [min(distances), max(distances)] if distances else None,
        "median_minimum_distance_mm": float(np.median(distances)) if distances else None,
        "hypothetical_thresholds": [
            {"distance_mm": t, "positive_studies": sum(len(x) for x in positive[t].values()),
             "complete_pairs": sum(len(x) == 2 for x in positive[t].values())}
            for t in thresholds
        ],
        "locked_threshold_mm": 10, "production_rule_changed": False,
        "outcome_data_read": False, "test_used": False, "segmentation_accuracy_measured": False,
    }
    atomic_write_json(root / "candidate_distance_diagnostic.json", result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
