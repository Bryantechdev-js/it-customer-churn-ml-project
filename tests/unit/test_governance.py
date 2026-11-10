"""Tests for aggregate model governance diagnostics."""

import numpy as np
import pytest

from churn_prediction.governance import subgroup_performance_report


def test_subgroup_report_computes_metrics_for_each_group():
    report = subgroup_performance_report(
        y_true=[0, 1, 0, 1],
        probabilities=[0.1, 0.8, 0.7, 0.4],
        group_columns={"segment": ["A", "A", "B", "B"]},
        threshold=0.5,
    )

    assert report["rows"].tolist() == [2, 2]
    assert report["group_value"].tolist() == ["A", "B"]
    assert report.loc[0, "recall"] == pytest.approx(1.0)
    assert report.loc[1, "false_positive_rate"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("y_true", "probabilities", "groups", "threshold"),
    [
        ([0, 1], [0.2], {"g": ["a", "b"]}, 0.5),
        ([0, 2], [0.2, 0.8], {"g": ["a", "b"]}, 0.5),
        ([0, 1], [np.nan, 0.8], {"g": ["a", "b"]}, 0.5),
        ([0, 1], [0.2, 0.8], {"g": ["a"]}, 0.5),
        ([0, 1], [0.2, 0.8], {"g": ["a", "b"]}, 1.1),
    ],
)
def test_subgroup_report_rejects_invalid_inputs(
    y_true,
    probabilities,
    groups,
    threshold,
):
    with pytest.raises(ValueError):
        subgroup_performance_report(
            y_true,
            probabilities,
            groups,
            threshold,
        )
