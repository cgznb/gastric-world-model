from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from stageworld.cli import app
from stageworld.errors import ConfigurationError
from stageworld.release_scan import scan_release_tree


def _write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def test_clean_release_scan_allows_placeholders_normative_text_and_pack_checksums(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "configs/project.yaml",
        "api_key: ${STAGEWORLD_API_KEY}\nclinical_excel: /path/to/clinical.xlsx\n",
    )
    _write(
        tmp_path / "docs/safety.md",
        (
            "Never commit credentials, bearer tokens, PHI, DICOM, WSI, NIfTI, or Excel assets.\n"
            "Generated files remain below <output>/data and ${OUTPUT_ROOT}/patient-artifacts.\n"
        ),
    )
    _write(tmp_path / "src/package.py", "token_count = 16\n")
    _write(tmp_path / "tests/test_fixture.py", "patient_id = 'SYNTHETIC-001'\n")
    _write(tmp_path / "artifacts/synthetic/report.json", '{"mode": "synthetic"}\n')
    _write(
        tmp_path / "PACK_MANIFEST.json",
        json.dumps({"sha256": "a" * 64}),
    )

    first = scan_release_tree(tmp_path)
    second = scan_release_tree(tmp_path)

    assert first.passed
    assert first.as_dict() == second.as_dict()
    assert first.as_dict()["finding_count"] == 0
    assert first.as_dict()["count_only"] is True
    assert first.as_dict()["matched_values_emitted"] is False
    assert first.as_dict()["file_paths_emitted"] is False


def test_release_scan_excludes_local_real_runtime_artifact_namespaces(
    tmp_path: Path,
) -> None:
    assigned_secret = "HighEntropy" + "ClinicalSecret" + "92741"
    _write(tmp_path / "artifacts/synthetic/report.json", '{"mode": "synthetic"}\n')
    _write(tmp_path / "artifacts/real/private/case.dcm", b"restricted runtime data")
    _write(
        tmp_path / "artifacts/real_audit/private.yaml",
        f"password = '{assigned_secret}'\n",
    )

    report = scan_release_tree(tmp_path)

    assert report.passed
    assert report.scanned_files == 1
    assert dict(report.scope_file_counts) == {"artifact": 1}


def test_release_scan_detects_secrets_sensitive_paths_and_raw_assets_without_values(
    tmp_path: Path,
) -> None:
    provider_token = "gh" + "p_" + ("A1" * 20)
    assigned_secret = "C0mplex" + "Credential" + "92741"
    aws_secret = "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3h2G1f0"
    sensitive_path = "/" + "root/autodl-tmp/" + "Datasets/private-clinical"
    private_key_header = "-----BEGIN " + "PRIVATE KEY-----"
    _write(
        tmp_path / "configs/private.yaml",
        f"api_key: '{provider_token}'\nAWS_SECRET_ACCESS_KEY={aws_secret}\n",
    )
    _write(tmp_path / "tests/fixture.py", f"password = '{assigned_secret}'\n")
    _write(tmp_path / "docs/run.md", f"local_source: {sensitive_path}\n")
    _write(tmp_path / "src/key.txt", private_key_header + "\n")
    _write(tmp_path / "artifacts/raw/case.dcm", b"patient bytes are never emitted")
    _write(tmp_path / "artifacts/raw/slide.svs", b"patient bytes are never emitted")
    _write(
        tmp_path / "artifacts/raw/pathology/section.tiff",
        b"patient bytes are never emitted",
    )
    _write(tmp_path / "artifacts/raw/volume.nii.gz", b"patient bytes are never emitted")
    _write(tmp_path / "artifacts/raw/table.xlsx", b"patient bytes are never emitted")
    _write(
        tmp_path / "artifacts/raw/disguised.bin",
        (b"\x00" * 128) + b"DICM" + b"patient bytes are never emitted",
    )

    report = scan_release_tree(tmp_path)
    payload = report.as_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert not report.passed
    counts = dict(report.finding_counts)
    assert counts["credential.github_token"] == 1
    assert counts["credential.assigned_secret"] == 2
    assert counts["credential.private_key"] == 1
    assert counts["sensitive_path.absolute_data"] == 1
    assert counts["raw_asset.dicom"] == 2
    assert counts["raw_asset.wsi"] == 2
    assert counts["raw_asset.nifti"] == 1
    assert counts["raw_asset.excel"] == 1
    assert payload["files_with_findings"] == 10
    for sensitive_value in (
        provider_token,
        assigned_secret,
        aws_secret,
        sensitive_path,
        "case.dcm",
    ):
        assert sensitive_value not in serialized
    assert str(tmp_path) not in serialized


def test_release_scan_fails_closed_for_unscanned_large_file(tmp_path: Path) -> None:
    _write(tmp_path / "artifacts/model.bin", b"x" * 513)

    report = scan_release_tree(tmp_path, content_limit_bytes=512)

    assert not report.passed
    assert dict(report.finding_counts) == {"scan.content_limit_exceeded": 1}
    assert report.as_dict()["scope_complete"] is False


def test_release_scan_blocks_unknown_top_level_directory_without_reading_it(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "unexpected/private/patient-record.txt", "must not be scanned")

    report = scan_release_tree(tmp_path)

    assert dict(report.finding_counts) == {"scan.unscoped_directory": 1}
    assert report.scanned_files == 0


def test_release_scan_fails_closed_when_walk_reports_unreadable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_walk(
        root: Path,
        *,
        followlinks: bool,
        onerror: Any,
    ) -> list[tuple[str, list[str], list[str]]]:
        assert root == tmp_path
        assert followlinks is False
        onerror(PermissionError(13, "permission denied", os.fspath(tmp_path / "artifacts")))
        return []

    monkeypatch.setattr("stageworld.release_scan.os.walk", failed_walk)

    report = scan_release_tree(tmp_path)

    assert not report.passed
    assert dict(report.finding_counts) == {"scan.unreadable_directory": 1}
    assert report.as_dict()["scope_complete"] is False


def test_release_scan_rejects_invalid_root_and_limit(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    _write(file_path, "content")
    with pytest.raises(ConfigurationError) as root_error:
        scan_release_tree(file_path)
    assert root_error.value.code == "INVALID_RELEASE_SCAN_ROOT"

    with pytest.raises(ConfigurationError) as limit_error:
        scan_release_tree(tmp_path, content_limit_bytes=511)
    assert limit_error.value.code == "INVALID_RELEASE_SCAN_LIMIT"


def test_release_scan_cli_uses_exit_status_as_release_gate(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean"
    _write(clean_root / "src/package.py", "value = 1\n")
    clean = CliRunner().invoke(app, ["release-scan", "--root", str(clean_root)])
    assert clean.exit_code == 0, clean.output
    assert json.loads(clean.output)["status"] == "ok"

    blocked_root = tmp_path / "blocked"
    _write(blocked_root / "artifacts/raw.svs", b"not emitted")
    blocked = CliRunner().invoke(app, ["release-scan", "--root", str(blocked_root)])
    assert blocked.exit_code == 2
    payload = json.loads(blocked.output)
    assert payload["status"] == "blocked"
    assert payload["finding_counts"] == {"raw_asset.wsi": 1}


def test_release_scan_cli_does_not_resolve_away_symlink_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    _write(real_root / "src/package.py", "value = 1\n")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    result = CliRunner().invoke(app, ["release-scan", "--root", str(linked_root)])

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "INVALID_RELEASE_SCAN_ROOT"
