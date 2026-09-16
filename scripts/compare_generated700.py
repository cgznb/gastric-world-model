"""Paired aggregate-only comparisons against completed original700 predictions."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.generated700_evaluation import points


def load_oof(root: Path, arm: str, seed: int) -> dict:
    return torch.load(
        root / "evaluation" / f"{arm}-{seed}-oof.pt", weights_only=True, map_location="cpu"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    pairs = [
        ("generated_v2_frozen_bce", "generated_bce", args.old),
        ("generated_v2_bce", "generated_bce", args.old),
        ("generated_v2_balanced", "generated_balanced", args.old),
        ("generated_v2_focal", "generated_focal", args.old),
        ("generated_v2_bce", "generated_v2_frozen_bce", args.new),
    ]
    draws = np.random.default_rng(1701).integers(0, 700, size=(1000, 700))
    results = {}
    for new_arm, old_arm, reference_root in pairs:
        new = [load_oof(args.new, new_arm, seed) for seed in (17, 43, 97)]
        old = [load_oof(reference_root, old_arm, seed) for seed in (17, 43, 97)]
        for current, previous in zip(new, old, strict=True):
            if (
                current["patient_ids"] != previous["patient_ids"]
                or len(current["patient_ids"]) != 700
            ):
                raise ValueError("Paired patient order/coverage differs")
            if any(not torch.equal(current[k], previous[k]) for k in ("labels", "valid")):
                raise ValueError("Paired labels or masks differ")
        result = {}
        for endpoint, name in enumerate(("pcr", "recurrence")):
            valid = new[0]["valid"][:, endpoint].numpy()
            labels = new[0]["labels"][:, endpoint].numpy()
            prepared = [
                [
                    (
                        row["probabilities"][:, endpoint].numpy(),
                        row["inner_balanced"][:, endpoint].numpy(),
                    )
                    for row in collection
                ]
                for collection in (new, old)
            ]

            def difference(
                indices: np.ndarray, valid=valid, labels=labels, prepared=prepared
            ) -> dict:
                rows = indices[valid[indices]]
                y = labels[rows]
                metrics = [
                    [points(p[rows], y, decision[rows]) for p, decision in collection]
                    for collection in prepared
                ]
                return {
                    metric: float(
                        np.mean([m[metric] for m in metrics[0]])
                        - np.mean([m[metric] for m in metrics[1]])
                    )
                    for metric in (
                        "auroc",
                        "auprc",
                        "sensitivity",
                        "specificity",
                        "balanced_accuracy",
                    )
                    if all(m[metric] is not None for group in metrics for m in group)
                }

            samples: dict[str, list[float]] = defaultdict(list)
            for draw in draws:
                for metric, value in difference(draw).items():
                    samples[metric].append(value)
            result[name] = {
                metric: {
                    "difference": value,
                    "ci95": np.quantile(samples[metric], [0.025, 0.975]).tolist()
                    if len(samples[metric]) >= 800
                    else None,
                    "valid_resamples": len(samples[metric]),
                }
                for metric, value in difference(np.arange(700)).items()
            }
        results[f"{new_arm}_minus_{old_arm}"] = result
    scale_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for path in args.new.glob("fits/generated*/*/fold-*/selection.json"):
        selected = read_json(path)["inner_residual_scales"]
        for endpoint, scale in zip(("pcr", "recurrence"), selected, strict=True):
            scale_counts[path.parents[2].name][endpoint][str(scale)] += 1
    ct: dict[str, list[dict]] = defaultdict(list)
    for path in args.new.glob("fits/generated*/*/fold-*/ct_evaluation.json"):
        ct[path.parents[2].name].append(read_json(path))
    ct_means = {
        arm: {
            control: {
                metric: float(
                    np.average(
                        [r[control][metric] for r in records],
                        weights=[r[control]["n"] for r in records],
                    )
                )
                for metric in ("mse", "smooth_l1_cosine")
            }
            for control in ("generated", "training_mean", "copy_ct0")
        }
        for arm, records in ct.items()
    }
    result = {
        "status": "completed",
        "matched_patients": 700,
        "seeds": [17, 43, 97],
        "bootstrap_replicates": 1000,
        "bootstrap_unit": "paired_patient_shared_across_seeds_fixed_OOF",
        "multiplicity_adjusted": False,
        "paired_comparisons": results,
        "inner_residual_scale_counts": scale_counts,
        "ct_feature_means": ct_means,
        "old_study_retrained": False,
        "external_validation": False,
    }
    atomic_write_private_json(args.new / "evaluation/architecture_comparison.json", result)
    print(
        {"status": "completed", "paired_comparisons": len(results), "bootstrap_replicates": 1000},
        flush=True,
    )


if __name__ == "__main__":
    main()
