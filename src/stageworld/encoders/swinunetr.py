"""Strict offline loader for the public Swin UNETR self-supervised CT backbone."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stageworld.errors import ArtifactError, DataContractError

from .medical import StateDictLoadReport, load_state_dict_strictly

SWINUNETR_SOURCE_VERSION = "monai-1.6.0-legacy-merging-v0.9-compatible-v1"
SWINUNETR_COMPONENT_VERSION = "swinunetr-ssl-model-swinvit-release-0.8.1"
SWINUNETR_PREPROCESS_VERSION = (
    "swinunetr-ct-ras-1p5x1p5x2-hu-m1000-p1000-center96-v1"
)
SWINUNETR_FEATURE_DIM = 768

_EXPECTED_TASK_HEAD_KEYS = frozenset(
    {
        "contrastive_head.bias",
        "contrastive_head.weight",
        "convTrans3d.bias",
        "convTrans3d.weight",
        "norm.bias",
        "norm.weight",
        "rotation_head.bias",
        "rotation_head.weight",
    }
)


class LegacySwinUNETRPatchMerging(nn.Module):
    """Reproduce the 3-D patch ordering used by the released SSL checkpoint.

    The historical implementation duplicated two offsets and omitted two others.
    Correcting that behavior while retaining the trained reduction matrices changes
    every deeper feature map, so compatibility requires preserving it here.
    """

    def __init__(
        self,
        dim: int,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            factor = 8
        elif spatial_dims == 2:
            factor = 4
        else:
            raise ValueError("LegacySwinUNETRPatchMerging supports only 2-D or 3-D inputs")
        self.reduction = nn.Linear(factor * dim, 2 * dim, bias=False)
        self.norm = norm_layer(factor * dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 5:
            _, depth, height, width, _ = x.shape
            if depth % 2 or height % 2 or width % 2:
                # This ordering intentionally mirrors the historical implementation.
                x = F.pad(
                    x,
                    (0, 0, 0, depth % 2, 0, width % 2, 0, height % 2),
                )
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 0::2, 1::2, 0::2, :]
            x6 = x[:, 0::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            merged = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), dim=-1)
        elif x.ndim == 4:
            _, height, width, _ = x.shape
            if height % 2 or width % 2:
                x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
            merged = torch.cat(
                (
                    x[:, 0::2, 0::2, :],
                    x[:, 1::2, 0::2, :],
                    x[:, 0::2, 1::2, :],
                    x[:, 1::2, 1::2, :],
                ),
                dim=-1,
            )
        else:
            raise ValueError(f"expected a 4-D or 5-D channels-last tensor, got {x.ndim}-D")
        return self.reduction(self.norm(merged))


class SwinUNETRDeepFeatureBackend(nn.Module):
    """Expose only the normalized deepest 768-channel backbone feature map."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1] != 1:
            raise DataContractError(
                code="INVALID_SWINUNETR_INPUT",
                message="Swin UNETR input must have shape [B,1,D,H,W].",
            )
        if any(int(size) % 32 for size in images.shape[2:]):
            raise DataContractError(
                code="INVALID_SWINUNETR_INPUT_SIZE",
                message="Every Swin UNETR spatial input dimension must be divisible by 32.",
                details={"spatial_shape": list(images.shape[2:])},
            )
        features = self.backbone(images, normalize=True)
        if not isinstance(features, Sequence) or len(features) != 5:
            raise DataContractError(
                code="INVALID_SWINUNETR_BACKBONE_OUTPUT",
                message="The Swin UNETR backbone must return five feature scales.",
            )
        deepest = features[-1]
        if not isinstance(deepest, Tensor) or deepest.ndim != 5:
            raise DataContractError(
                code="INVALID_SWINUNETR_BACKBONE_OUTPUT",
                message="The deepest Swin UNETR feature must be [B,768,D,H,W].",
            )
        if deepest.shape[0] != images.shape[0] or deepest.shape[1] != SWINUNETR_FEATURE_DIM:
            raise DataContractError(
                code="INVALID_SWINUNETR_BACKBONE_OUTPUT",
                message="The deepest Swin UNETR feature has an incompatible shape.",
            )
        return deepest


@dataclass(frozen=True)
class SwinUNETRLoadResult:
    backend: SwinUNETRDeepFeatureBackend
    state_dict_report: StateDictLoadReport
    excluded_task_head_keys: tuple[str, ...]
    runtime_version: str


def map_swinunetr_ssl_state_dict(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    """Map the release checkpoint to current MONAI names with an exact head allowlist."""

    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ArtifactError(
            code="SWINUNETR_STATE_DICT_MISSING",
            message="The Swin UNETR checkpoint has no nonempty state_dict mapping.",
        )
    mapped: dict[str, Tensor] = {}
    excluded: set[str] = set()
    for raw_name, value in raw_state.items():
        if not isinstance(raw_name, str) or not isinstance(value, Tensor):
            raise ArtifactError(
                code="SWINUNETR_STATE_DICT_INVALID",
                message="The Swin UNETR state_dict must contain named tensors only.",
            )
        if not raw_name.startswith("module."):
            raise ArtifactError(
                code="SWINUNETR_KEY_PREFIX_MISMATCH",
                message="Every release checkpoint tensor must use the declared module prefix.",
            )
        name = raw_name.removeprefix("module.")
        if name in _EXPECTED_TASK_HEAD_KEYS:
            excluded.add(name)
            continue
        name = name.replace(".mlp.fc1.", ".mlp.linear1.")
        name = name.replace(".mlp.fc2.", ".mlp.linear2.")
        if name in mapped:
            raise ArtifactError(
                code="SWINUNETR_KEY_COLLISION",
                message="Checkpoint key mapping produced a duplicate backbone tensor.",
                details={"key": name},
            )
        mapped[name] = value
    if excluded != _EXPECTED_TASK_HEAD_KEYS:
        raise ArtifactError(
            code="SWINUNETR_TASK_HEAD_CONTRACT_MISMATCH",
            message="The checkpoint task-head tensors differ from the exact release allowlist.",
            details={
                "missing": sorted(_EXPECTED_TASK_HEAD_KEYS - excluded),
                "unexpected": sorted(excluded - _EXPECTED_TASK_HEAD_KEYS),
            },
        )
    return mapped, tuple(sorted(excluded))


def load_swinunetr_ssl_backbone(
    weight_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> SwinUNETRLoadResult:
    """Load the approved local checkpoint without network or executable remote code."""

    source = Path(weight_path).expanduser().resolve()
    if not source.is_file():
        raise ArtifactError(
            code="ENCODER_WEIGHTS_MISSING",
            message="The configured local Swin UNETR weight file is missing.",
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ArtifactError(
            code="SWINUNETR_CHECKPOINT_UNREADABLE",
            message="The local Swin UNETR checkpoint is unreadable in weights-only mode.",
        ) from error
    if not isinstance(payload, Mapping):
        raise ArtifactError(
            code="SWINUNETR_CHECKPOINT_INVALID",
            message="The Swin UNETR checkpoint root must be a mapping.",
        )

    try:
        import einops  # noqa: F401
        import monai
        from monai.networks.nets.swin_unetr import SwinTransformer
    except ImportError as error:
        raise ArtifactError(
            code="ENCODER_DEPENDENCY_MISSING",
            message="Swin UNETR requires MONAI and einops in the local CT environment.",
        ) from error
    if monai.__version__ != "1.6.0":
        raise ArtifactError(
            code="SWINUNETR_RUNTIME_VERSION_MISMATCH",
            message="The validated Swin UNETR adapter requires MONAI 1.6.0 exactly.",
            details={"expected": "1.6.0", "actual": monai.__version__},
        )

    backbone = SwinTransformer(
        in_chans=1,
        embed_dim=48,
        window_size=(7, 7, 7),
        patch_size=(2, 2, 2),
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        patch_norm=False,
        use_checkpoint=False,
        spatial_dims=3,
        downsample=LegacySwinUNETRPatchMerging,
        use_v2=False,
    )
    mapped, excluded = map_swinunetr_ssl_state_dict(payload)
    report = load_state_dict_strictly(backbone, mapped)
    if (
        report.missing_keys
        or report.unexpected_keys
        or report.shape_mismatches
        or report.loaded_parameter_fraction != 1.0
    ):
        raise ArtifactError(
            code="SWINUNETR_BACKBONE_COVERAGE_FAILED",
            message="Swin UNETR backbone loading did not achieve exact tensor coverage.",
        )
    backend = SwinUNETRDeepFeatureBackend(backbone)
    backend.requires_grad_(False)
    backend.eval()
    backend.to(torch.device(device))
    return SwinUNETRLoadResult(
        backend=backend,
        state_dict_report=report,
        excluded_task_head_keys=excluded,
        runtime_version=f"monai-{monai.__version__}",
    )


__all__ = [
    "LegacySwinUNETRPatchMerging",
    "SWINUNETR_COMPONENT_VERSION",
    "SWINUNETR_FEATURE_DIM",
    "SWINUNETR_PREPROCESS_VERSION",
    "SWINUNETR_SOURCE_VERSION",
    "SwinUNETRDeepFeatureBackend",
    "SwinUNETRLoadResult",
    "load_swinunetr_ssl_backbone",
    "map_swinunetr_ssl_state_dict",
]
