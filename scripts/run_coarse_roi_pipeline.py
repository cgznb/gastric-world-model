"""Host-local coarse ROI extraction, treatment preparation and 100-epoch OS training."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.errors import ArtifactError


def _workbook_candidates(roots: set[Path]):
    excluded = {".venv", ".git", "node_modules", "third_party"}
    seen = set()
    for root in sorted(roots):
        for directory, names, files in os.walk(root):
            relative = Path(directory).relative_to(root)
            names[:] = [name for name in names if name not in excluded]
            if len(relative.parts) >= 8:
                names[:] = []
            for name in files:
                if name.lower().endswith(".xlsx") and not name.startswith("~$"):
                    path = (Path(directory) / name).resolve()
                    if path not in seen:
                        seen.add(path)
                        yield path


def prepare_environment(project: Path, output: Path) -> tuple[dict[str, str], dict]:
    from stageworld.data.paired_ct import HMACPseudonymizer
    from stageworld.data.treatment_regimens import read_regimen_rows
    from stageworld.data.tumor_roi import validate_tumor_model
    from stageworld.real_workflow import _select_feature_bindings

    private = project / "artifacts/real/paired-ct-os-v1-first-acquisition/data/restricted"
    assets, splits = (read_json(private / name) for name in (
        "asset_bindings.json", "split_assignments.json",
    ))
    ct_root = Path(os.path.commonpath([row["local_path"] for row in assets["bindings"]]))
    selected = _select_feature_bindings(assets, splits, limit_per_split=None, include_test=False)
    patients = {str(row["patient_id"]) for row, _ in selected}
    environment = os.environ.copy()
    environment.pop("STAGEWORLD_WEIAI_ROOT", None)
    bindings_path = output / "restricted/runtime_bindings.json"
    saved = read_json(bindings_path) if bindings_path.exists() else {}
    for name, filename in (
        ("STAGEWORLD_HMAC_KEY_FILE", "gastric-os-v1.hmac"),
        ("STAGEWORLD_SWINUNETR_WEIGHT", "model_swinvit.pt"),
    ):
        bound = environment.get(name) or saved.get(name)
        matches = [Path(bound)] if bound else list(Path.home().rglob(filename))
        if len(matches) != 1 or not matches[0].is_file():
            raise ArtifactError(code="HOST_BINDING_UNAVAILABLE", message="Resolve local assets.")
        environment[name] = str(matches[0].resolve())
    pseudonymizer = HMACPseudonymizer.from_file(
        Path(environment["STAGEWORLD_HMAC_KEY_FILE"]), project_root=project
    )
    prior = read_json(
        Path.home() / ".local/share/stageworld/gastric-regimen-os-v1/treatment_manifest.json"
    )
    prior_patients = {row["patient_id"] for row in prior["rows"]}
    bound_workbook = environment.get("STAGEWORLD_CLINICAL_EXCEL") or saved.get(
        "STAGEWORLD_CLINICAL_EXCEL"
    )
    search_roots = {ct_root.parent, Path.home()}
    search_roots.update(
        path.resolve() for path in Path.home().iterdir() if path.is_symlink() and path.is_dir()
    )
    candidates = [Path(bound_workbook)] if bound_workbook else _workbook_candidates(search_roots)
    matched = []
    development_rows = None
    discovery: Counter[str] = Counter()
    for candidate in candidates:
        discovery["workbooks_examined"] += 1
        try:
            previous_rows = read_regimen_rows(
                candidate, patient_ids=prior_patients, pseudonymizer=pseudonymizer
            )
            if previous_rows != prior["rows"]:
                discovery["previous_snapshot_mismatch"] += 1
                continue
            current_rows = read_regimen_rows(
                candidate, patient_ids=patients, pseudonymizer=pseudonymizer
            )
        except Exception as error:
            discovery[str(getattr(error, "code", type(error).__name__))] += 1
            continue
        if development_rows is not None and current_rows != development_rows:
            raise ArtifactError(
                code="GROUPED_WORKBOOK_AMBIGUOUS", message="Matching sources disagree."
            )
        development_rows = current_rows
        matched.append(candidate)
        discovery["sources_reproducing_previous_snapshot"] += 1
    atomic_write_private_json(output / "workbook_discovery.json", dict(discovery))
    if not matched:
        raise ArtifactError(
            code="CONFIRMED_GROUPED_WORKBOOK_UNAVAILABLE",
            message="The source must reproduce the previously confirmed treatment snapshot.",
        )
    environment["STAGEWORLD_CLINICAL_EXCEL"] = str(matched[0].resolve())
    feature_root = Path.home() / ".local/share/stageworld/flare23-tumor-coarse-fallback-os-v1"
    environment["STAGEWORLD_TUMOR_ROI_FEATURE_ROOT"] = str(feature_root)
    environment["STAGEWORLD_TUMOR_TREATMENT_MANIFEST"] = str(
        feature_root / "treatment_manifest.json"
    )
    environment["STAGEWORLD_TUMOR_MODEL"] = str(
        Path.home() / ".cache/stageworld-models/flare23-blackbean-20260911"
    )
    validate_tumor_model(Path(environment["STAGEWORLD_TUMOR_MODEL"]))
    environment.update(nnUNet_compile="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    names = (
        "STAGEWORLD_HMAC_KEY_FILE", "STAGEWORLD_SWINUNETR_WEIGHT", "STAGEWORLD_CLINICAL_EXCEL",
        "STAGEWORLD_TUMOR_ROI_FEATURE_ROOT", "STAGEWORLD_TUMOR_TREATMENT_MANIFEST",
        "STAGEWORLD_TUMOR_MODEL",
    )
    atomic_write_private_json(bindings_path, {name: environment[name] for name in names})
    summary = {
        "status": "ok", "development_patients": len(patients), "selected_studies": len(selected),
        "previous_treatment_rows_reproduced": len(prior_patients),
        "development_treatment_rows_available": len(development_rows or []),
        "outcome_data_read": False, "test_features_used": False,
        "segmentation_labels_required": False,
    }
    atomic_write_private_json(output / "preflight.json", summary)
    return environment, summary


def verify_source_snapshot(project: Path, output: Path) -> None:
    snapshot = output / "source_snapshot"
    record = snapshot / "files.json"
    if record.exists():
        files = read_json(record)["files"]
        if any((project / name).read_bytes() != (snapshot / name).read_bytes() for name in files):
            raise ArtifactError(code="PIPELINE_SOURCE_CHANGED", message="Active source changed.")
        return
    files = ["pyproject.toml"]
    for directory, suffix in (("src", ".py"), ("scripts", ".py"), ("configs", ".yaml")):
        files.extend(str(path.relative_to(project)) for path in sorted((project / directory).rglob(
            f"*{suffix}"
        )))
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(project / name, target)
        target.chmod(0o600)
    atomic_write_private_json(record, {"files": files, "comparison": "full_byte_content"})


def verify_pilot_crops(environment: dict[str, str], output: Path) -> dict:
    import nibabel as nib
    import numpy as np

    root = Path(environment["STAGEWORLD_TUMOR_ROI_FEATURE_ROOT"])
    manifest = read_json(root / "ct_manifest.json")
    assert manifest["test_features_included"] is False
    masks = root / "tumor_masks"
    counts: Counter[str] = Counter()
    roles = defaultdict(set)
    for row in manifest["entries"]:
        asset = row["asset_id"]
        full = nib.load(masks / f"{asset}.nii.gz")
        crop = nib.load(masks / f"{asset}.crop.nii.gz")
        candidate = nib.load(masks / f"{asset}.candidate.nii.gz")
        qc = read_json(masks / f"{asset}.qc.json")
        labels, binary, values = (np.asarray(image.dataobj) for image in (full, candidate, crop))
        assert row["split"] in ("train", "validation")
        roles[row["patient_id"]].add(row["role"])
        assert crop.shape == (96, 96, 96) and np.isfinite(values).all()
        assert float(np.ptp(values)) > 1 and bool((values > -800).any())
        assert np.array_equal(candidate.affine, full.affine)
        assert np.isin(binary, (0, 1)).all() and np.all(labels[binary > 0] == 14)
        source = qc["roi_source"]
        assert source == row["roi_source"] and qc["outside_mask_ct_preserved"]
        selected = binary > 0 if source == "tumor_candidate" else labels == 11
        assert bool(selected.any())
        positions = np.argwhere(selected)
        low, high = positions.min(0) - 0.5, positions.max(0) + 0.5
        corners = np.array([
            [x, y, z, 1] for x in (low[0], high[0])
            for y in (low[1], high[1]) for z in (low[2], high[2])
        ]) @ full.affine.T
        spacing = np.diag(crop.affine)[:3]
        assert (spacing > 0).all() and np.allclose(spacing, spacing[0])
        crop_low = crop.affine[:3, 3] - spacing / 2
        crop_high = crop_low + 96 * spacing
        margin = 30 if source == "tumor_candidate" else 40
        assert np.all(corners[:, :3].min(0) - crop_low >= margin - 1e-3)
        assert np.all(crop_high - corners[:, :3].max(0) >= margin - 1e-3)
        counts[source] += 1
    unsafe = sum(bool(path.stat().st_mode & 0o077) for path in masks.rglob("*"))
    assert unsafe == 0
    assert len(manifest["entries"]) == 8
    assert len(roles) == 4
    assert all(value == {"baseline_ct", "post_treatment_ct"} for value in roles.values())
    result = {
        "status": "ok", "crops_checked": len(manifest["entries"]),
        "nonblank_finite_crops": len(manifest["entries"]), "roi_source_counts": dict(counts),
        "physical_margin_checks_passed": True, "unsafe_permissions": unsafe,
        "test_features_used": False, "segmentation_accuracy_measured": False,
    }
    atomic_write_private_json(output / "pilot_verification.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "pilot", "run"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project.resolve()
    os.chdir(project)
    os.umask(0o077)
    output = project / "artifacts/real/flare23-tumor-coarse-fallback-regimen-os-100ep-v1"
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    state = {"mode": args.mode, "pid": os.getpid(), "status": "running", "stage": "preflight"}

    def save() -> None:
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        state["elapsed_seconds"] = time.monotonic() - started
        atomic_write_private_json(output / "pipeline_progress.json", state)

    with (output / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "failed", "error_code": "PIPELINE_ALREADY_RUNNING"}))
            return 2
        with (output / "restricted_pipeline.log").open("a") as log:
            try:
                save()
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    environment, result = prepare_environment(project, output)
                if args.mode != "preflight":
                    phases = ("extract",) if args.mode == "pilot" else (
                        "extract", "prepare-treatments", "train", "verify",
                    )
                    for phase in phases:
                        if args.mode == "run":
                            verify_source_snapshot(project, output)
                        state.update(stage=phase, status="running")
                        save()
                        command = [sys.executable, "scripts/run_local_gastric_os.py", phase,
                                   "--coarse-tumor-roi"]
                        if args.mode == "pilot":
                            command += ["--limit-per-split", "2"]
                        phase_environment = environment.copy()
                        if phase == "extract":
                            phase_environment.pop("STAGEWORLD_CLINICAL_EXCEL", None)
                        completed = subprocess.run(
                            command, cwd=project, env=phase_environment, stdout=log,
                            stderr=subprocess.STDOUT, check=False,
                        )
                        state["last_exit_code"] = completed.returncode
                        if completed.returncode != 0:
                            raise ArtifactError(
                                code="PIPELINE_PHASE_FAILED",
                                message="Inspect the private phase log.",
                            )
                        if phase == "extract":
                            extracted = read_json(output / "features/ct_summary.json")
                            result = {
                                "status": extracted["status"],
                                "requested_studies": extracted["requested_study_count"],
                                "complete_pairs": extracted[
                                    "complete_patient_pair_counts_by_split"
                                ],
                                "roi_sources": extracted["roi_source_counts_before_pair_filter"],
                                "failure_counts": extracted["failure_counts"],
                            }
                            if args.mode == "run" and extracted["engineering_smoke_only"]:
                                raise ArtifactError(
                                    code="FULL_EXTRACTION_REQUIRED",
                                    message="Complete full extraction.",
                                )
                        if phase == "verify":
                            trained = read_json(output / "os_summary.json")
                            result.update(
                                world_completed_epochs=trained["pretraining_completed_epochs"],
                                joint_completed_epochs=trained["joint_completed_epochs"],
                                training_patients=trained["training_patients"],
                                validation_patients=trained["validation_patients"],
                                verification_exit_code=completed.returncode,
                            )
                    if args.mode == "pilot":
                        result["crop_verification"] = verify_pilot_crops(environment, output)
                state.update(status="completed", stage="completed", result=result)
                save()
                print(json.dumps(state))
                return 0
            except BaseException as error:
                traceback.print_exc(file=log)
                state.update(
                    status="failed", error_code=getattr(error, "code", type(error).__name__)
                )
                save()
                print(json.dumps(state))
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
