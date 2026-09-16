from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import torch

from stageworld.cache import CacheProvenance, FeatureCache
from stageworld.encoders import EncoderProvenance, ObservationTokens
from stageworld.errors import ArtifactError


def encoder_provenance(*, frozen: bool = True) -> EncoderProvenance:
    return EncoderProvenance(
        encoder_name="unit_encoder",
        source_version="official-v1",
        component_versions=(("backbone", "weights-v1"),),
        preprocess_version="preprocess-v1",
        feature_dim=3,
        frozen_source=frozen,
    )


def cache_provenance(**changes: object) -> CacheProvenance:
    values: dict[str, object] = {
        "schema_version": "schema-v1",
        "encoder_name": "unit_encoder",
        "encoder_source_version": "official-v1",
        "component_versions": (("backbone", "weights-v1"),),
        "preprocess_version": "preprocess-v1",
        "patch_sampling_version": "patch-v1",
        "split_version": "split-v1",
        "target_transform_version": "target-v1",
        "teacher_version": "teacher-v1",
        "feature_dim": 3,
        "frozen_source": True,
    }
    values.update(changes)
    return CacheProvenance(**values)  # type: ignore[arg-type]


def tokens(*, requires_grad: bool = False) -> ObservationTokens:
    return ObservationTokens(
        values=torch.arange(6.0).reshape(1, 2, 3).requires_grad_(requires_grad),
        valid=torch.tensor([[True, False]]),
        modality=torch.tensor([[0, 0]]),
        acquired_time=torch.tensor([[0.0, 0.0]]),
        available_time=torch.tensor([[1.0, 0.0]]),
        provenance=encoder_provenance(),
        source_id=(("ct-a", "padding"),),
        modality_name="ct",
    )


def test_atomic_cache_complete_round_trip(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    provenance = cache_provenance()
    decision = cache.begin("patient-001", provenance)
    assert decision.state == "incomplete"
    cache.store("patient-001", provenance, tokens())
    loaded = cache.load("patient-001", provenance)
    assert torch.equal(loaded.values, tokens().values)
    assert loaded.modality_name == "ct"
    status = json.loads((tmp_path / "patient-001" / "status.json").read_text())
    assert status["state"] == "complete"
    assert "checksum" not in json.dumps(status).lower()


def test_cache_directories_and_lock_are_private(tmp_path: Path) -> None:
    root = tmp_path / "shared-looking-cache"
    root.mkdir(mode=0o755)
    cache = FeatureCache(root)

    cache.begin("private-entry", cache_provenance())

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "private-entry").stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "private-entry" / ".lock").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "field",
    [
        "encoder_source_version",
        "preprocess_version",
        "patch_sampling_version",
        "split_version",
        "target_transform_version",
        "teacher_version",
    ],
)
def test_any_changed_version_invalidates_cache(tmp_path: Path, field: str) -> None:
    cache = FeatureCache(tmp_path)
    original = cache_provenance()
    cache.begin("entry", original)
    cache.store("entry", original, tokens())
    changed = cache_provenance(**{field: "changed-v2"})
    with pytest.raises(ArtifactError) as exc:
        cache.load("entry", changed)
    assert exc.value.code == "STALE_CACHE_PROVENANCE"
    refreshed = cache.begin("entry", changed)
    assert not refreshed.already_complete
    assert not refreshed.resume


def test_incomplete_and_failed_records_resume(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    provenance = cache_provenance()
    assert not cache.begin("entry", provenance).resume
    assert cache.begin("entry", provenance).resume
    cache.mark_failed("entry", provenance, failure_code="UPSTREAM_QC_FAILED")
    with pytest.raises(ArtifactError) as exc:
        cache.load("entry", provenance)
    assert exc.value.code == "CACHE_NOT_COMPLETE"
    assert cache.begin("entry", provenance).resume


def test_get_or_compute_receives_resume_flag(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    provenance = cache_provenance()
    cache.begin("entry", provenance)
    seen: list[bool] = []
    loaded = cache.get_or_compute(
        "entry", provenance, lambda resume: seen.append(resume) or tokens()
    )
    assert seen == [True]
    assert loaded.provenance.encoder_name == "unit_encoder"


def test_half_written_and_corrupt_payloads_are_never_read(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    provenance = cache_provenance()
    directory = tmp_path / "half"
    directory.mkdir()
    torch.save(tokens().as_cache_payload(), directory / "tokens.pt")
    with pytest.raises(ArtifactError) as exc:
        cache.load("half", provenance)
    assert exc.value.code == "HALF_WRITTEN_CACHE"

    cache.begin("corrupt", provenance)
    cache.store("corrupt", provenance, tokens())
    (tmp_path / "corrupt" / "tokens.pt").write_bytes(b"not-a-torch-payload")
    with pytest.raises(ArtifactError) as exc:
        cache.load("corrupt", provenance)
    assert exc.value.code == "CORRUPT_CACHE_PAYLOAD"


@pytest.mark.parametrize("entry_id", ["../escape", "nested/path", "..", "", "/absolute"])
def test_cache_rejects_path_traversal(tmp_path: Path, entry_id: str) -> None:
    cache = FeatureCache(tmp_path)
    with pytest.raises(ArtifactError) as exc:
        cache.begin(entry_id, cache_provenance())
    assert exc.value.code == "INVALID_CACHE_ENTRY_ID"


def test_cache_rejects_online_tensor_and_non_frozen_provenance(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    provenance = cache_provenance()
    cache.begin("online", provenance)
    with pytest.raises(ArtifactError) as exc:
        cache.store("online", provenance, tokens(requires_grad=True))
    assert exc.value.code == "ONLINE_FEATURE_CACHE_FORBIDDEN"
    with pytest.raises(ArtifactError) as exc:
        cache.begin("unfrozen", cache_provenance(frozen_source=False))
    assert exc.value.code == "ONLINE_FEATURE_CACHE_FORBIDDEN"


def test_digest_only_versions_are_rejected_without_echoing_value() -> None:
    secret_like_digest = "abcdef0123456789"
    with pytest.raises(Exception) as exc:
        cache_provenance(encoder_source_version=secret_like_digest)
    assert exc.value.code == "UNNAMED_ENCODER_VERSION"
    assert secret_like_digest not in str(exc.value)
