"""Spreadsheet-safe CSV download regression tests."""

import pandas as pd
import pytest

from churn_prediction.exporting import safe_csv_bytes


def test_csv_export_neutralizes_formula_like_text_without_mutating_input():
    source = pd.DataFrame(
        {
            "reference": ["=2+2", "\t=1+1", "Normal", " -1"],
            "numeric": [-1, 2, 3, 4],
        }
    )
    original = source.copy(deep=True)

    exported = safe_csv_bytes(source).decode("utf-8")

    assert "'=2+2" in exported
    assert "'\t=1+1" in exported
    assert "Normal" in exported
    assert source.equals(original)
    assert "-1" in exported


def test_csv_export_requires_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        safe_csv_bytes([{"cell": "=1+1"}])
