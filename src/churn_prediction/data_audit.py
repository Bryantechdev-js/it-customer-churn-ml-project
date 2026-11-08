
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "it_customer_churn.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "initial_data_audit.json"


EXPECTED_COLUMNS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
    "Customer type",
    "Custom value",
    "Index",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def audit_dataset() -> dict:
    """Validate the initial dataset and produce an audit report."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check the filename and data/raw directory."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Schema validation
    actual_columns = set(df.columns)
    missing_columns = sorted(EXPECTED_COLUMNS - actual_columns)
    unexpected_columns = sorted(actual_columns - EXPECTED_COLUMNS)

    # Target validation
    if "Churn" not in df.columns:
        raise ValueError("Required target column 'Churn' is missing.")

    if df["Churn"].isna().any():
        raise ValueError("Target column 'Churn' contains missing values.")

    target_values = set(df["Churn"].unique())
    invalid_target_values = sorted(
        str(value) for value in target_values - {0, 1}
    )

    if invalid_target_values:
        raise ValueError(
            f"Invalid Churn values found: {invalid_target_values}"
        )

    # Dataset quality checks
    duplicate_rows = int(df.duplicated().sum())

    missing_values = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    # Target distribution
    target_counts = {
        str(label): int(count)
        for label, count in df["Churn"].value_counts().sort_index().items()
    }

    target_percentages = {
        str(label): round(float(percentage), 4)
        for label, percentage in (
            df["Churn"].value_counts(normalize=True).sort_index() * 100
        ).items()
    }

    # Check whether Customer type is derived from tenure.
    customer_type_check = {"checked": False, "matches": None}

    if {"Customer type", "tenure"}.issubset(df.columns):
        expected_customer_type = np.where(
            pd.to_numeric(df["tenure"], errors="coerce") <= 39,
            "New",
            "Old",
        )

        customer_type_check = {
            "checked": True,
            "matches": bool(
                np.array_equal(
                    df["Customer type"].astype(str).to_numpy(),
                    expected_customer_type,
                )
            ),
        }

    # Check whether Custom value is derived from TotalCharges.
    custom_value_check = {"checked": False, "matches": None}

    if {"Custom value", "TotalCharges"}.issubset(df.columns):
        total_charges = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

        expected_custom_value = np.where(
            total_charges >= 5000,
            "High",
            "Low",
        )

        custom_value_check = {
            "checked": True,
            "matches": bool(
                np.array_equal(
                    df["Custom value"].astype(str).to_numpy(),
                    expected_custom_value,
                )
            ),
        }

    # Check whether Index is a sequential row number.
    index_check = {"checked": False, "matches": None}

    if "Index" in df.columns:
        index_values = pd.to_numeric(df["Index"], errors="coerce")

        index_check = {
            "checked": True,
            "matches": bool(
                index_values.equals(
                    pd.Series(
                        range(1, len(df) + 1),
                        index=df.index,
                        dtype="int64",
                    )
                )
            ),
        }

    # Overall structural validation.
    schema_valid = not missing_columns and not unexpected_columns

    report = {
        "audit_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "filename": DATA_PATH.name,
            "sha256": calculate_sha256(DATA_PATH),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "duplicate_rows": duplicate_rows,
        },
        "schema": {
            "valid": schema_valid,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "actual_columns": list(df.columns),
        },
        "missing_values": missing_values,
        "target": {
            "column": "Churn",
            "valid_binary_target": True,
            "class_counts": target_counts,
            "class_percentages": target_percentages,
        },
        "feature_checks": {
            "customer_type_derived_from_tenure": customer_type_check,
            "custom_value_derived_from_total_charges": custom_value_check,
            "index_is_sequential": index_check,
        },
        "limitations": [
            "No observation timestamp is available.",
            "The 30-day future churn label definition is unverified.",
            "Dataset provenance and licence require verification.",
            "InternetService missing values require investigation.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("DATA AUDIT COMPLETED")
    print("--------------------")
    print(f"Dataset: {DATA_PATH.name}")
    print(f"Shape: {df.shape}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Schema valid: {schema_valid}")
    print(f"Missing values: {missing_values}")
    print(f"Target distribution: {target_counts}")
    print(f"Report saved to: {REPORT_PATH}")

    if not schema_valid:
        raise ValueError(
            "Schema validation failed. Inspect the audit report."
        )

    return report


if __name__ == "__main__":
    audit_dataset()
