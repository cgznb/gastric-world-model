"""Count DICOM annotation storage classes within the approved CT tree; no pixel reads."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pydicom
from pydicom.filereader import read_file_meta_info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    os.umask(0o077)
    assets = json.loads(
        (
            project
            / (
                "artifacts/real/paired-ct-os-v1-first-acquisition/data/restricted/asset_bindings.json"
            )
        ).read_text()
    )
    approved_root = Path(os.path.commonpath([x["local_path"] for x in assets["bindings"]]))
    output = project / "artifacts/real/flare23-gastric-tumor-pilot-v1/annotation_header_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    classes = {
        "1.2.840.10008.5.1.4.1.1.2": "ct_image_storage",
        "1.2.840.10008.5.1.4.1.1.2.1": "enhanced_ct_image_storage",
        "1.2.840.10008.5.1.4.1.1.66.4": "dicom_seg",
        "1.2.840.10008.5.1.4.1.1.66.5": "surface_segmentation",
        "1.2.840.10008.5.1.4.1.1.66.7": "labelmap_segmentation",
        "1.2.840.10008.5.1.4.1.1.66.8": "heightmap_segmentation",
        "1.2.840.10008.5.1.4.1.1.481.3": "rt_structure_set",
    }
    counts: Counter[str] = Counter()
    files = directories = 0
    started = time.monotonic()

    def classify(path: Path) -> str:
        if path.suffix.lower() != ".dcm":
            return "non_dicom_extension"
        try:
            meta = read_file_meta_info(str(path))
            storage_class = str(meta.get("MediaStorageSOPClassUID", ""))
            if not storage_class:
                header = pydicom.dcmread(
                    path, stop_before_pixels=True, specific_tags=["SOPClassUID"]
                )
                storage_class = str(header.get("SOPClassUID", ""))
            if storage_class in classes:
                return classes[storage_class]
            known = pydicom.uid.UID_dictionary.get(storage_class)
            return "other: " + known[0] if known else "unrecognized_storage_class"
        except Exception:
            return "unreadable_header"

    from stageworld.artifacts import atomic_write_private_json

    def scan_error(error: OSError) -> None:
        counts["directory_scan_errors"] += 1

    last_progress = started
    with ThreadPoolExecutor(max_workers=64) as pool:
        for directory, _, names in os.walk(approved_root, onerror=scan_error):
            directories += 1
            paths = [Path(directory) / name for name in names]
            files += len(paths)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                counts.update(pool.map(classify, paths))
            if time.monotonic() - last_progress > 30:
                progress = {
                    "status": "running",
                    "directories": directories,
                    "files": files,
                    "counts": dict(counts),
                }
                atomic_write_private_json(output, progress)
                print(json.dumps(progress), flush=True)
                last_progress = time.monotonic()
    result = {
        "status": "completed" if not counts["directory_scan_errors"] else "incomplete",
        "scope": "approved_CT_asset_common_root",
        "directories": directories,
        "files": files,
        "counts": dict(counts),
        "elapsed_seconds": time.monotonic() - started,
        "pixel_data_read": False,
        "outcome_data_read": False,
        "limitation": (
            "Storage classes only; private overlays and external roots are not adjudicated."
        ),
    }
    atomic_write_private_json(output, result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
