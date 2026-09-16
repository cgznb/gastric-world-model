from __future__ import annotations

from pathlib import Path

import pytest

from stageworld.config import RunMode, load_config
from stageworld.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_config_is_typed_and_valid() -> None:
    config = load_config(ROOT / "configs/project.synthetic.yaml")
    assert config.mode is RunMode.SYNTHETIC
    assert config.model.hidden_dim == 32
    config.validate(command="train", supervised=True)


def test_real_supervised_training_requires_signoff(monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = ROOT / "tests"
    monkeypatch.setenv("STAGEWORLD_WEIAI_ROOT", str(data_root))
    config = load_config(ROOT / "configs/project.weiai-audit.yaml")
    assert config.mode is RunMode.REAL_IMAGES
    with pytest.raises(ConfigurationError, match="unsigned clinical fields") as exc:
        config.validate(command="train", supervised=True)
    assert exc.value.code == "CLINICAL_SIGNOFF_REQUIRED"
    assert "os_event_mapping" in exc.value.details["missing"]


def test_real_audit_allows_unsigned_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = ROOT / "tests"
    monkeypatch.setenv("STAGEWORLD_WEIAI_ROOT", str(data_root))
    config = load_config(ROOT / "configs/project.weiai-audit.yaml")
    config.validate(command="audit", supervised=False)


def test_unsafe_options_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        "project: {mode: synthetic}\nprivacy: {allow_external_tracking: true}\n",
        encoding="utf-8",
    )
    config = load_config(path)
    with pytest.raises(ConfigurationError) as exc:
        config.validate(command="doctor")
    assert exc.value.code == "UNSAFE_CONFIGURATION"
