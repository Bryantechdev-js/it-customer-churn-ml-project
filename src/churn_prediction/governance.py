"""Descriptive subgroup evaluation helpers for model review."""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def subgroup_performance_report(
    y_true,
    probabilities,
    group_columns,
    threshold,
):
    """Return aggregate error and calibration metrics for review groups.

    This report is diagnostic only. It does not establish fairness,
    statistical significance, or legal compliance.
    """

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional.")
    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Targets and probabilities must have equal non-zero length.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Targets must contain only 0 and 1.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0).any() or (probabilities > 1).any()
    ):
        raise ValueError("Probabilities must be finite values from 0 to 1.")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    predictions = (probabilities >= float(threshold)).astype(int)
    rows = []

    for column, values in group_columns.items():
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != len(y_true):
            raise ValueError(f"Group '{column}' must align with the target rows.")

        for group in pd.unique(values):
            mask = pd.isna(values) if pd.isna(group) else values == group
            labels = y_true[mask]
            scores = probabilities[mask]
            flags = predictions[mask]
            tn, fp, fn, tp = confusion_matrix(
                labels,
                flags,
                labels=[0, 1],
            ).ravel()

            rows.append(
                {
                    "group_attribute": str(column),
                    "group_value": str(group),
                    "rows": int(mask.sum()),
                    "observed_churn_rate": float(labels.mean()),
                    "predicted_flag_rate": float(flags.mean()),
                    "precision": float(
                        precision_score(labels, flags, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(labels, flags, zero_division=0)
                    ),
                    "false_positive_rate": (
                        float(fp / (fp + tn)) if fp + tn else np.nan
                    ),
                    "false_negative_rate": (
                        float(fn / (fn + tp)) if fn + tp else np.nan
                    ),
                    "roc_auc": (
                        float(roc_auc_score(labels, scores))
                        if len(np.unique(labels)) == 2
                        else np.nan
                    ),
                    "pr_auc": float(average_precision_score(labels, scores)),
                    "brier_score": float(brier_score_loss(labels, scores)),
                }
            )

    return pd.DataFrame(rows)
