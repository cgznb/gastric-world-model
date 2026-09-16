from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stageworld.cli import app
from stageworld.errors import ResourceError

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_cli_is_redacted(tmp_path: Path) -> None:
    config_text = (ROOT / "configs/project.synthetic.yaml").read_text(encoding="utf-8")
    config_text = config_text.replace("artifacts/synthetic/demo", str(tmp_path / "run"))
    config = tmp_path / "config.yaml"
    config.write_text(config_text, encoding="utf-8")
    result = CliRunner().invoke(app, ["doctor", "--config", str(config)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["raw_paths_redacted"] is True
    assert "clinical_excel" not in result.output
    report = Path(payload["report"])
    assert report.is_file()


def test_train_cli_serializes_structured_resource_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_with_oom(*args: object, **kwargs: object) -> None:
        raise ResourceError(
            code="CUDA_OUT_OF_MEMORY",
            message="Synthetic accelerator exhaustion.",
            remediation="Reduce the patient microbatch and resume from a checkpoint.",
        )

    monkeypatch.setattr("stageworld.cli.run_synthetic_training", fail_with_oom)
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--phase",
            "world_pretrain",
            "--config",
            str(ROOT / "configs/project.synthetic.yaml"),
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "CUDA_OUT_OF_MEMORY"
    assert "microbatch" in payload["error"]["remediation"]


def test_real_build_cli_wires_signed_paired_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "approved-data"
    data_root.mkdir()
    workbook = data_root / "clinical.xlsx"
    workbook.touch()
    key = tmp_path / "identity.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    monkeypatch.setenv("STAGEWORLD_WEIAI_ROOT", str(data_root))
    monkeypatch.setenv("STAGEWORLD_CLINICAL_EXCEL", str(workbook))
    monkeypatch.setenv("STAGEWORLD_HMAC_KEY_FILE", str(key))
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_build(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    def fake_write(result: object, **kwargs: object) -> dict[str, object]:
        assert result is sentinel
        captured.update(kwargs)
        return {"status": "ok", "included_patient_count": 7}

    monkeypatch.setattr("stageworld.cli.build_paired_ct_cohort", fake_build)
    monkeypatch.setattr("stageworld.cli.write_paired_ct_artifacts", fake_write)
    result = CliRunner().invoke(
        app,
        [
            "build-cohort",
            "--config",
            str(ROOT / "configs/project.weiai-os-v1.yaml"),
            "--dicom-workers",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["included_patient_count"] == 7
    assert captured["workers"] == 3
    assert captured["workbook_path"] == str(workbook)


@pytest.mark.parametrize("command", ["train", "predict", "evaluate"])
def test_roi_os_cli_routes_authorized_development_protocol(tmp_path, monkeypatch, command):
    monkeypatch.setenv("STAGEWORLD_WEIAI_ROOT", str(tmp_path))
    name = {
        "train": "run_real_os_development",
        "predict": "predict_real_os_development",
        "evaluate": "evaluate_real_os_development",
    }[command]
    monkeypatch.setattr(f"stageworld.real_survival.{name}", lambda *a, **kw: {"status": "routed"})
    args = [command, "--config", str(ROOT / "configs/project.weiai-os-v1-gastric-roi.yaml")]
    if command == "train":
        args += ["--phase", "joint_survival"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "routed"
