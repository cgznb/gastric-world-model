import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from stageworld.artifacts import read_json
from stageworld.errors import ArtifactError


@pytest.fixture
def pipeline():
    path = Path(__file__).resolve().parents[1] / "scripts/run_coarse_roi_pipeline.py"
    spec = importlib.util.spec_from_file_location("coarse_pipeline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_extraction_stops_before_treatment_or_training(tmp_path, monkeypatch, pipeline):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline.os, "umask", lambda _: 0o077)
    monkeypatch.setattr(pipeline.sys, "argv", ["pipeline", "run", "--project", str(tmp_path)])
    monkeypatch.setattr(pipeline, "prepare_environment", lambda *a: (
        {"STAGEWORLD_CLINICAL_EXCEL": "synthetic-workbook.xlsx"}, {"status": "ok"}
    ))
    monkeypatch.setattr(pipeline, "verify_source_snapshot", lambda *a: None)
    calls = []

    def fail(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    assert pipeline.main() == 1
    assert len(calls) == 1 and calls[0][0][2:] == ["extract", "--coarse-tumor-roi"]
    assert "STAGEWORLD_CLINICAL_EXCEL" not in calls[0][1]
    result = read_json(tmp_path / (
        "artifacts/real/flare23-tumor-coarse-fallback-regimen-os-100ep-v1/pipeline_progress.json"
    ))
    assert result["status"] == "failed" and result["stage"] == "extract"
    assert result["last_exit_code"] == 7


def test_pipeline_source_snapshot_rejects_later_code_changes(tmp_path, pipeline):
    (tmp_path / "pyproject.toml").write_text("synthetic-project")
    output = tmp_path / "output"
    pipeline.verify_source_snapshot(tmp_path, output)
    pipeline.verify_source_snapshot(tmp_path, output)
    (tmp_path / "pyproject.toml").write_text("changed-project")
    with pytest.raises(ArtifactError) as error:
        pipeline.verify_source_snapshot(tmp_path, output)
    assert error.value.code == "PIPELINE_SOURCE_CHANGED"
