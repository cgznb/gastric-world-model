from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from stageworld.config import RunMode
from stageworld.encoders import (
    CTGeometry,
    EncoderAccess,
    EncoderProvenance,
    EncoderRegistry,
    MerlinEncoder,
    ObservationTokens,
    PathologyPatchBatch,
    PRISM2Encoder,
    SyntheticEncoder,
    TITANCONCHEncoder,
    WSIGeometry,
    load_state_dict_strictly,
)
from stageworld.errors import ArtifactError, ConfigurationError, DataContractError, StageWorldError


def access(
    mode: RunMode,
    *,
    components: dict[str, str],
    weights: dict[str, Path] | None = None,
    **changes: object,
) -> EncoderAccess:
    values: dict[str, object] = {
        "mode": mode,
        "source_version": "official-release-v1",
        "component_versions": components,
        "preprocess_version": "official-preprocess-v1",
        "version_approved": True,
        "license_name": "approved-research-license",
        "license_approved": True,
        "weight_paths": weights or {},
    }
    values.update(changes)
    return EncoderAccess(**values)  # type: ignore[arg-type]


def geometry() -> CTGeometry:
    return CTGeometry(
        spacing_mm=torch.ones(1, 3),
        origin_mm=torch.zeros(1, 3),
        direction=torch.eye(3)[None],
        spatial_shape=torch.tensor([[4, 4, 4]]),
    )


def patches(name: str, dim: int, *, mpp: float = 0.5, size: int = 224) -> PathologyPatchBatch:
    wsi = WSIGeometry(
        coords_level0=torch.tensor([[[0.0, 0.0], [224.0, 0.0]]]),
        valid=torch.tensor([[True, True]]),
        mpp=torch.full((1, 2), mpp),
        patch_size_level0=torch.tensor([float(size)]),
    )
    return PathologyPatchBatch(
        features=torch.ones(1, 2, dim),
        geometry=wsi,
        patch_encoder_name=name,
        patch_encoder_version="approved-patch-v1",
        input_patch_size_px=size,
        magnification=20.0,
        slide_ids=("slide-a",),
        tissue_sources=("unknown",),
        acquired_time=torch.tensor([1.0]),
        available_time=torch.tensor([2.0]),
    )


def test_synthetic_encoder_is_deterministic_and_mode_gated() -> None:
    encoder = SyntheticEncoder(mode=RunMode.SYNTHETIC, input_dim=2, feature_dim=3)
    kwargs = {
        "valid": torch.tensor([[True, True]]),
        "modality": torch.zeros(1, 2, dtype=torch.long),
        "acquired_time": torch.zeros(1, 2),
        "available_time": torch.zeros(1, 2),
        "source_id": (("a", "b"),),
        "modality_name": "ct",
    }
    first = encoder.encode(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), **kwargs)
    second = encoder.encode(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), **kwargs)
    assert torch.equal(first.observations.values, second.observations.values)
    with pytest.raises(ConfigurationError) as exc:
        SyntheticEncoder(mode=RunMode.REAL_FEATURES, input_dim=2, feature_dim=3)
    assert exc.value.code == "SYNTHETIC_ENCODER_FORBIDDEN"


def test_registry_rejects_duplicates_and_unknown_names() -> None:
    registry = EncoderRegistry()
    registry.register("synthetic", SyntheticEncoder)
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register("SYNTHETIC", SyntheticEncoder)
    with pytest.raises(ConfigurationError) as exc:
        registry.create("missing")
    assert exc.value.code == "UNKNOWN_ENCODER"


def test_real_image_gate_fails_before_backend_call(tmp_path: Path) -> None:
    called = False

    def backend(_: torch.Tensor) -> torch.Tensor:
        nonlocal called
        called = True
        return torch.ones(1, 2048)

    encoder = MerlinEncoder(
        access=access(RunMode.REAL_IMAGES, components={"merlin": "merlin-v1"}),
        backend=backend,
    )
    with pytest.raises(ArtifactError) as exc:
        encoder.encode(
            torch.zeros(1, 1, 4, 4, 4),
            geometry=geometry(),
            source_ids=("ct-a",),
            acquired_time=torch.tensor([0.0]),
            available_time=torch.tensor([0.0]),
        )
    assert exc.value.code == "ENCODER_WEIGHTS_MISSING"
    assert not called


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"network_enabled": True}, "ENCODER_NETWORK_FORBIDDEN"),
        ({"version_approved": False}, "ENCODER_VERSION_NOT_APPROVED"),
        ({"license_approved": False}, "ENCODER_LICENSE_NOT_APPROVED"),
        ({"source_version": "abcdef0123456789"}, "UNNAMED_ENCODER_VERSION"),
    ],
)
def test_real_image_gate_rejects_unauthorized_configuration(
    tmp_path: Path, changes: dict[str, object], code: str
) -> None:
    weight = tmp_path / "merlin.pt"
    weight.write_bytes(b"local-test-placeholder")
    encoder = MerlinEncoder(
        access=access(
            RunMode.REAL_IMAGES,
            components={"merlin": "merlin-v1"},
            weights={"merlin": weight},
            **changes,
        ),
        backend=lambda _: torch.ones(1, 2048),
    )
    with pytest.raises(StageWorldError) as exc:
        encoder.encode(
            torch.zeros(1, 1, 4, 4, 4),
            geometry=geometry(),
            source_ids=("ct-a",),
            acquired_time=torch.tensor([0.0]),
            available_time=torch.tensor([0.0]),
        )
    assert exc.value.code == code


def test_merlin_global_output_stays_global_not_fake_spatial(tmp_path: Path) -> None:
    weight = tmp_path / "merlin.pt"
    weight.write_bytes(b"local-test-placeholder")
    encoder = MerlinEncoder(
        access=access(
            RunMode.REAL_IMAGES,
            components={"merlin": "merlin-v1"},
            weights={"merlin": weight},
        ),
        backend=lambda _: torch.ones(1, 2048),
    )
    output = encoder.encode(
        torch.zeros(1, 1, 4, 4, 4),
        geometry=geometry(),
        source_ids=("ct-a",),
        acquired_time=torch.tensor([0.0]),
        available_time=torch.tensor([0.0]),
    )
    assert output.observations.values.shape == (1, 1, 2048)
    assert output.observations.coords is None


def test_merlin_genuine_spatial_map_has_physical_coordinates(tmp_path: Path) -> None:
    weight = tmp_path / "merlin.pt"
    weight.write_bytes(b"local-test-placeholder")
    encoder = MerlinEncoder(
        access=access(
            RunMode.REAL_IMAGES,
            components={"merlin": "merlin-v1"},
            weights={"merlin": weight},
        ),
        feature_dim=4,
        backend=lambda _: {
            "global_embedding": torch.ones(1, 2048),
            "spatial_features": torch.ones(1, 4, 2, 2, 2),
        },
    )
    output = encoder.encode(
        torch.zeros(1, 1, 4, 4, 4),
        geometry=geometry(),
        source_ids=("ct-a",),
        acquired_time=torch.tensor([0.0]),
        available_time=torch.tensor([0.0]),
    )
    assert output.observations.values.shape == (1, 8, 4)
    assert output.observations.coordinate_system == "patient_physical_mm"


def test_titan_rejects_uni_features_before_backend(tmp_path: Path) -> None:
    titan = tmp_path / "titan.pt"
    conch = tmp_path / "conch.pt"
    titan.write_bytes(b"t")
    conch.write_bytes(b"c")
    remote = tmp_path / "audited-code"
    remote.mkdir()
    called = False

    def backend(*_: object) -> torch.Tensor:
        nonlocal called
        called = True
        return torch.ones(1, 768)

    encoder = TITANCONCHEncoder(
        access=access(
            RunMode.REAL_IMAGES,
            components={"titan": "titan-v1", "conch": "conch-v1.5"},
            weights={"titan": titan, "conch": conch},
            remote_code_path=remote,
            remote_code_approved=True,
        ),
        backend=backend,
    )
    with pytest.raises(DataContractError) as exc:
        encoder.encode(patches("uni2_h", 1536))
    assert exc.value.code == "INCOMPATIBLE_PATCH_ENCODER"
    assert not called


@pytest.mark.parametrize(
    ("patch_name", "dim", "mpp", "size", "code"),
    [
        ("uni2_h", 1280, 0.5, 224, "INCOMPATIBLE_PATCH_ENCODER"),
        ("virchow2_cls", 768, 0.5, 224, "INVALID_FEATURE_DIMENSION"),
        ("virchow2_cls", 1280, 1.0, 224, "INVALID_MPP"),
        ("virchow2_cls", 1280, 0.5, 256, "INVALID_PATCH_PROTOCOL"),
    ],
)
def test_prism2_rejects_wrong_input_protocol(
    tmp_path: Path, patch_name: str, dim: int, mpp: float, size: int, code: str
) -> None:
    prism = tmp_path / "prism.pt"
    virchow = tmp_path / "virchow.pt"
    prism.write_bytes(b"p")
    virchow.write_bytes(b"v")
    remote = tmp_path / "audited-code"
    remote.mkdir()
    encoder = PRISM2Encoder(
        access=access(
            RunMode.REAL_IMAGES,
            components={"prism2": "prism2-v1", "virchow2": "virchow2-v1"},
            weights={"prism2": prism, "virchow2": virchow},
            remote_code_path=remote,
            remote_code_approved=True,
        ),
        backend=lambda *_: torch.ones(1, 2560),
    )
    with pytest.raises(DataContractError) as exc:
        encoder.encode(patches(patch_name, dim, mpp=mpp, size=size))
    assert exc.value.code == code


def test_feature_only_requires_exact_identity_and_frozen_values() -> None:
    encoder = MerlinEncoder(
        access=access(RunMode.REAL_FEATURES, components={"merlin": "merlin-v1"})
    )
    compatible = ObservationTokens(
        values=torch.ones(1, 1, 2048),
        valid=torch.ones(1, 1, dtype=torch.bool),
        modality=torch.zeros(1, 1, dtype=torch.long),
        acquired_time=torch.zeros(1, 1),
        available_time=torch.zeros(1, 1),
        provenance=encoder.provenance,
        source_id=(("ct-a",),),
        modality_name="ct",
    )
    assert encoder.validate_precomputed(compatible).observations is compatible
    incompatible = ObservationTokens(
        values=torch.ones(1, 1, 2048),
        valid=torch.ones(1, 1, dtype=torch.bool),
        modality=torch.zeros(1, 1, dtype=torch.long),
        acquired_time=torch.zeros(1, 1),
        available_time=torch.zeros(1, 1),
        provenance=EncoderProvenance(
            encoder_name="other",
            source_version="official-release-v1",
            component_versions=(("merlin", "merlin-v1"),),
            preprocess_version="official-preprocess-v1",
            feature_dim=2048,
        ),
        source_id=(("ct-a",),),
        modality_name="ct",
    )
    with pytest.raises(DataContractError) as exc:
        encoder.validate_precomputed(incompatible)
    assert exc.value.code == "ENCODER_PROVENANCE_MISMATCH"


def test_state_dict_loader_reports_full_coverage() -> None:
    source = nn.Linear(3, 2)
    target = nn.Linear(3, 2)
    report = load_state_dict_strictly(target, source.state_dict())
    assert report.loaded_parameter_fraction == 1.0
    assert not report.missing_keys
    assert all(
        torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters(), strict=True)
    )


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape"])
def test_state_dict_loader_rejects_incomplete_or_incompatible_weights(kind: str) -> None:
    module = nn.Linear(3, 2)
    state = dict(module.state_dict())
    if kind == "missing":
        state.pop("bias")
    elif kind == "unexpected":
        state["other"] = torch.ones(1)
    else:
        state["weight"] = torch.ones(3, 2)
    with pytest.raises(ArtifactError) as exc:
        load_state_dict_strictly(module, state)
    assert exc.value.code == "STATE_DICT_COVERAGE_FAILED"
    assert "loaded_parameter_fraction" in exc.value.details
