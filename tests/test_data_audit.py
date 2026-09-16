from __future__ import annotations

import json
from datetime import datetime

import pytest

from stageworld.data import ExcelAuditConfig, audit_excel, write_redacted_excel_audit

openpyxl = pytest.importorskip("openpyxl")


def test_redacted_excel_audit_handles_double_headers_types_and_invalid_dates(tmp_path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "synthetic-sheet"
    worksheet.append(["Identity", "Clinical", None, "Outcome", "Derived"])
    worksheet.append(["patient name", "assessment_date", "assessment_date", "status", "calc"])
    worksheet.append(["SECRET-PERSON-A", datetime(2025, 1, 1), "not-a-date", 0, "=1+1"])
    worksheet.append(["SECRET-PERSON-B", "#N/A", None, 7, None])
    source = tmp_path / "synthetic.xlsx"
    workbook.save(source)

    report = audit_excel(
        source,
        ExcelAuditConfig(
            header_rows=(1, 2),
            date_columns=frozenset({"B", "C"}),
            field_encodings={"D": frozenset({"0", "1"})},
        ),
    )

    assert report.record_count == 2
    assert report.column_count == 5
    assert report.duplicate_header_count == 1
    assert report.formula_cell_count == 1
    assert report.error_cell_count == 1
    by_letter = {column.source_column_letter: column for column in report.columns}
    assert by_letter["A"].source_header == "[REDACTED_IDENTIFIER_FIELD]"
    assert by_letter["B"].source_header_group == "Clinical"
    assert by_letter["B"].duplicate_header
    assert by_letter["C"].invalid_date_count == 1
    assert by_letter["D"].unexpected_encoding_count == 1
    assert by_letter["E"].formula_count == 1

    serialized = json.dumps(report.as_dict(), sort_keys=True)
    assert "SECRET-PERSON" not in serialized
    assert "not-a-date" not in serialized
    assert "=1+1" not in serialized


def test_redacted_audit_writer_never_persists_cell_values(tmp_path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["patient name", "safe_measurement"])
    worksheet.append(["SECRET-PERSON", 123456789])
    source = tmp_path / "synthetic.xlsx"
    target = tmp_path / "audit.json"
    workbook.save(source)

    report = audit_excel(source, ExcelAuditConfig())
    write_redacted_excel_audit(report, target)
    written = target.read_text(encoding="ascii")
    assert "SECRET-PERSON" not in written
    assert "123456789" not in written
    assert "[REDACTED_IDENTIFIER_FIELD]" in written
