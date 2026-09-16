"""Original700 admission, read-only frozen CT loading and fold-local transforms."""

from __future__ import annotations

import filecmp
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from stageworld.artifacts import atomic_write_private_json, new_artifact_id, read_json
from stageworld.binary_data import make_binary_folds, read_binary_labels
from stageworld.cache import CacheProvenance
from stageworld.data.baseline_clinical import (
    CT6_CLINICAL_SCHEMA,
    encode_baseline,
    fit_clinical_transform,
    read_baseline_rows,
)
from stageworld.data.paired_ct import HMACPseudonymizer
from stageworld.data.treatment_compact import encode_compact, fit_name_support, read_compact_rows
from stageworld.data.tumor_roi import TUMOR_MODEL_FILES
from stageworld.encoders.base import EncoderProvenance, ObservationTokens
from stageworld.model.compact_residual_binary import spatial_order
from stageworld.synthetic_workflow import _atomic_torch_save


@dataclass
class Pool:
    ids: list[str]
    clinical: dict[str, Any]
    treatments: dict[str, Any]
    interval: torch.Tensor
    ct0: torch.Tensor
    ct0_valid: torch.Tensor
    ct1: torch.Tensor
    ct1_valid: torch.Tensor
    labels: torch.Tensor
    valid: torch.Tensor
    artifact_id: str
    ct1_tokens: torch.Tensor

    def indices(self, ids: list[str]) -> torch.Tensor:
        lookup = {p: i for i, p in enumerate(self.ids)}
        return torch.tensor([lookup[p] for p in ids], dtype=torch.long)


def load_pool(output: Path) -> Pool:
    return Pool(**torch.load(output / "pool.pt", weights_only=True, map_location="cpu"))


def prepare_pool(source: Path, test_features: Path, bindings_path: Path, output: Path) -> Pool:
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output / "pool.pt").exists():
        return load_pool(output)
    original = source / "artifacts/real/paired-ct-os-v1-first-acquisition/data/restricted"
    cohort, split, assets = (
        read_json(original / name)
        for name in ("input_cohort.json", "split_assignments.json", "asset_bindings.json")
    )
    assignments = {r["patient_id"]: r["split"] for r in split["assignments"]}
    ids = sorted(assignments)
    if len(ids) != 700 or len(split["assignments"]) != 700 or cohort["outcomes"]:
        raise ValueError("Require the original700 input-only cohort")
    bindings = read_json(bindings_path)
    roots = (Path(bindings["STAGEWORLD_TUMOR_ROI_FEATURE_ROOT"]), test_features)
    manifests = [read_json(root / "ct_manifest.json") for root in roots]
    for manifest in manifests:
        if (
            manifest["cohort_artifact_id"] != cohort["cohort_artifact_id"]
            or manifest["data_lineage_id"] != cohort["data_lineage_id"]
            or manifest["split_version"] != split["split_version"]
            or manifest["outcome_data_read"]
        ):
            raise ValueError("Frozen feature lineage differs")
    if manifests[0]["encoder_provenance"] != manifests[1]["encoder_provenance"]:
        raise ValueError("Development and former-holdout CT encoders differ")
    provenance = [CacheProvenance.from_dict(m["cache_provenance"]) for m in manifests]
    comparable = []
    snapshots = [root / "tumor_masks/model_snapshot" for root in roots]
    for prov, snapshot in zip(provenance, snapshots, strict=True):
        binding = read_json(snapshot / "binding.json")
        expected_patch = f"{prov.preprocess_version}:{binding['model_artifact_id']}"
        if (
            prov.patch_sampling_version != expected_patch
            or binding["preprocess_version"] != prov.preprocess_version
        ):
            raise ValueError("CT sampling is not bound to its recorded segmenter")
        comparable.append({**prov.as_dict(), "patch_sampling_version": prov.preprocess_version})
    filecmp.clear_cache()
    if comparable[0] != comparable[1] or any(
        not filecmp.cmp(snapshots[0] / name, snapshots[1] / name, shallow=False)
        for name in TUMOR_MODEL_FILES
    ):
        raise ValueError("CT encoder, preprocessing, or exact segmentation model differs")
    encoder = EncoderProvenance.from_dict(manifests[0]["encoder_provenance"])
    if encoder.feature_dim != 768 or not encoder.frozen_source:
        raise ValueError("Use the verified frozen 768-dimensional Swin features")
    row_index = {p: i for i, p in enumerate(ids)}
    ct0, ct1 = torch.zeros(700, 27, 768), torch.zeros(700, 768)
    ct1_tokens = torch.zeros_like(ct0)
    valid0, valid1 = torch.zeros(700, dtype=torch.bool), torch.zeros(700, dtype=torch.bool)
    failures: Counter[str] = Counter()
    roles: Counter[tuple[str, str]] = Counter()
    inspected: list[dict[str, Any]] = []
    for row in assets["bindings"]:
        patient, role = row["patient_id"], row["role"]
        if patient not in row_index or role not in ("baseline_ct", "post_treatment_ct"):
            raise ValueError("Unexpected patient or image role")
        roles[patient, role] += 1
        group = int(assignments[patient] == "test")
        entry = roots[group] / "ct_entries" / row["asset_id"]
        status_path = entry / "status.json"
        status = read_json(status_path)
        if CacheProvenance.from_dict(status["provenance"]) != provenance[group]:
            raise ValueError("A cached CT has incompatible provenance")
        if status["state"] != "complete":
            if status.get("failure_code") != "COARSE_ANATOMIC_ROI_UNAVAILABLE":
                raise ValueError("Unexpected missing or incomplete image feature")
            failures[role] += 1
            continue
        path = entry / "tokens.pt"
        token = ObservationTokens.from_cache_payload(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        if token.provenance != encoder or token.values.shape != (1, 27, 768):
            raise ValueError("Cached CT tensor contract differs")
        order = spatial_order(token)
        value = token.values.gather(1, order[..., None].expand_as(token.values))[0].float()
        i = row_index[patient]
        if role == "baseline_ct":
            ct0[i], valid0[i] = value, True
        else:
            ct1[i], valid1[i] = value.mean(0), True
            ct1_tokens[i] = value
        inspected.append(
            {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        )
    if len(roles) != 1400 or set(roles.values()) != {1} or sum(failures.values()) != 3:
        raise ValueError("Original700 CT availability accounting differs")
    times = {(r["patient_id"], r["stage"]): r["query_time_days"] for r in cohort["queries"]}
    interval = torch.tensor([times[p, "s1"] - times[p, "s0"] for p in ids]).float()
    if not torch.isfinite(interval).all() or not (interval > 0).all():
        raise ValueError("Invalid declared target interval")
    pseudonymizer = HMACPseudonymizer.from_file(
        Path(bindings["STAGEWORLD_HMAC_KEY_FILE"]), project_root=source
    )
    workbook = Path(bindings["STAGEWORLD_CLINICAL_EXCEL"])
    clinical, clinical_audit = read_baseline_rows(
        workbook, set(ids), pseudonymizer, schema_version=CT6_CLINICAL_SCHEMA
    )
    treatments = {r["patient_id"]: r for r in read_compact_rows(workbook, set(ids), pseudonymizer)}
    labels = read_binary_labels(workbook, set(ids), pseudonymizer)
    folds = make_binary_folds(labels, seed=17, folds=5)
    if any(
        [len(f["patient_ids"][k]) for k in ("train", "validation", "outer")] != [448, 112, 140]
        for f in folds["folds"]
    ):
        raise ValueError("The requested700 folds do not have448/112/140 patients")
    pool = Pool(
        ids,
        clinical,
        treatments,
        interval,
        ct0,
        valid0,
        ct1,
        valid1,
        torch.tensor([[0 if v is None else v for v in labels[p]] for p in ids]).float(),
        torch.tensor([[v is not None for v in labels[p]] for p in ids]),
        new_artifact_id("generated700-inputs"),
        ct1_tokens,
    )
    audit = {
        "patients": 700,
        "label_counts": folds["pool"],
        "original_split_counts": dict(Counter(assignments.values())),
        "ct0_available": int(valid0.sum()),
        "ct1_available": int(valid1.sum()),
        "complete_ct_pairs": int((valid0 & valid1).sum()),
        "image_failures_by_role": dict(failures),
        "patients_excluded": 0,
        "clinical": clinical_audit,
        "former_holdout_merged": True,
        "cohort_origin": "original700_not_full956_source_records",
        "feature_reextraction": False,
        "cross_cache_segmenter_bytes_identical": True,
        "ct1_target": "unordered_27x768_frozen_feature_set",
    }
    atomic_write_private_json(output / "folds.json", folds)
    atomic_write_private_json(output / "audit.json", audit)
    atomic_write_private_json(output / "source_files.json", {"files": inspected})
    _atomic_torch_save(output / "pool.pt", vars(pool))
    return pool


def raw_tabular(pool: Pool, transform: dict, support: dict) -> np.ndarray:
    baseline = encode_baseline([pool.clinical[p] for p in pool.ids], transform)
    action, _ = encode_compact([pool.treatments[p] for p in pool.ids], support)
    return (
        torch.cat(
            (
                baseline.ridge_features(),
                action.flatten(1),
                torch.log1p(pool.interval[:, None] / 30),
            ),
            1,
        )
        .double()
        .numpy()
    )


def fit_inputs(pool: Pool, fitting: list[str], path: Path) -> tuple[torch.Tensor, dict]:
    indices = pool.indices(fitting)
    if not fitting or len(set(fitting)) != len(fitting):
        raise ValueError("Bind a unique nonempty fitting partition")
    if path.exists():
        snapshot = torch.load(path, weights_only=True, map_location="cpu")
        if snapshot["fit_ids"] != fitting or snapshot["pool_id"] != pool.artifact_id:
            raise ValueError("Input transform recovery differs")
    else:
        transform = fit_clinical_transform(
            pool.clinical, set(fitting), schema_version=CT6_CLINICAL_SCHEMA
        )
        support = fit_name_support(list(pool.treatments.values()), set(fitting))
        raw = raw_tabular(pool, transform, support)
        scaler = StandardScaler().fit(raw[indices.numpy()])
        snapshot = {
            "artifact_id": new_artifact_id("generated700-transform"),
            "pool_id": pool.artifact_id,
            "fit_ids": fitting,
            "clinical": transform,
            "support": support,
            "mean": torch.from_numpy(scaler.mean_),
            "scale": torch.from_numpy(scaler.scale_),
        }
        _atomic_torch_save(path, snapshot)
    raw = raw_tabular(pool, snapshot["clinical"], snapshot["support"])
    standardized = (raw - snapshot["mean"].numpy()) / snapshot["scale"].numpy()
    if not np.isfinite(standardized).all():
        raise ValueError("Nonfinite tabular input")
    return torch.from_numpy(standardized).float(), snapshot
