"""Redacted runtime diagnostics used by the local doctor command."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .config import StageWorldConfig


@dataclass(frozen=True)
class DependencyState:
    name: str
    available: bool
    purpose: str


def _nearest_existing(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def doctor_report(config: StageWorldConfig) -> dict[str, Any]:
    dependencies = [
        DependencyState("torch", True, "core model"),
        DependencyState("openpyxl", _present("openpyxl"), "clinical workbook audit"),
        DependencyState("pyarrow", _present("pyarrow"), "real standardized parquet tables"),
        DependencyState("pydicom", _present("pydicom"), "DICOM metadata and loading"),
        DependencyState("monai", _present("monai"), "medical image transforms/backbones"),
        DependencyState("openslide", _present("openslide"), "whole-slide image loading"),
        DependencyState("imagecodecs", _present("imagecodecs"), "TIFF codec support"),
        DependencyState("h5py", _present("h5py"), "TRIDENT-compatible feature IO"),
        DependencyState("pycox", _present("pycox"), "survival reference comparator"),
        DependencyState("sksurv", _present("sksurv"), "survival evaluation reference"),
        DependencyState("ruff", shutil.which("ruff") is not None, "linting"),
        DependencyState("mypy", shutil.which("mypy") is not None, "type checking"),
    ]
    disk = shutil.disk_usage(_nearest_existing(config.output_root))
    gpu: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu.update(
            {
                "device_name": properties.name,
                "memory_total_mib": round(properties.total_memory / 1024**2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return {
        "schema_version": "1.0",
        "mode": config.mode.value,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "gpu": gpu,
        "disk": {
            "total_gib": round(disk.total / 1024**3, 1),
            "free_gib": round(disk.free / 1024**3, 1),
        },
        "dependencies": [asdict(item) for item in dependencies],
        "clinical_blockers": (
            [] if config.mode.value == "synthetic" else config.clinical_blockers()
        ),
        "safety_findings": config.safety_findings(),
        "network_during_training": False,
        "external_tracking": False,
        "raw_paths_redacted": True,
        "executable": Path(sys.executable).name,
    }


def _present(module: str) -> bool:
    return importlib.util.find_spec(module) is not None
