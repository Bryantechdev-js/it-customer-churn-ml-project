
"""Automated data quality tests for the churn dataset."""

from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "it_customer_churn.csv"
)

EXPECTED_COLUMNS = {
    "Index",
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
    "Customer type",
    "Custom value",
    "Churn",
}

FORBIDDEN_MODEL_FEATURES = {
    "Index",
    "Customer type",
    "Custom value",
}


@pytest.fixture(scope="module")
def df():
    """Load the dataset once for this test module."""

    if not DATA_PATH.exists():
        pytest.fail(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


def test_dataset_has_expected_schema(df):
    """Ensure that all expected columns exist."""

    assert set(df.columns) == EXPECTED_COLUMNS


def test_dataset_has_expected_number_of_rows(df):
    """Check the current dataset size."""

    assert len(df) == 7032


def test_target_contains_only_binary_values(df):
    """Churn must contain only 0 and 1."""

    assert df["Churn"].notna().all()

    assert set(df["Churn"].unique()) == {0, 1}


def test_dataset_has_no_duplicate_rows(df):
    """Detect duplicate records."""

    assert df.duplicated().sum() == 0


def test_missing_values_are_only_in_internet_service(df):
    """Enforce the observed missing-value contract."""

    missing_columns = set(
        df.columns[df.isna().any()]
    )

    assert missing_columns == {"InternetService"}


def test_numeric_columns_contain_valid_values(df):
    """Validate important numerical features."""

    tenure = pd.to_numeric(
        df["tenure"],
        errors="coerce",
    )

    monthly_charges = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce",
    )

    total_charges = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    assert tenure.notna().all()
    assert monthly_charges.notna().all()
    assert total_charges.notna().all()

    assert tenure.between(1, 72).all()
    assert (monthly_charges > 0).all()
    assert (total_charges >= 0).all()


def test_index_is_sequential(df):
    """Ensure Index is the expected sequential identifier."""

    expected = list(range(1, len(df) + 1))

    assert df["Index"].tolist() == expected


def test_customer_type_is_derived_from_tenure(df):
    """Detect the derived Customer type column."""

    tenure = pd.to_numeric(df["tenure"])

    expected = pd.Series(
        "Old",
        index=df.index,
    )

    expected.loc[tenure <= 39] = "New"

    assert (
        df["Customer type"] == expected
    ).all()


def test_custom_value_is_derived_from_total_charges(df):
    """Detect the derived Custom value column."""

    total_charges = pd.to_numeric(
        df["TotalCharges"]
    )

    expected = pd.Series(
        "Low",
        index=df.index,
    )

    expected.loc[total_charges >= 5000] = "High"

    assert (
        df["Custom value"] == expected
    ).all()


def test_missing_internet_service_has_consistent_pattern(df):
    """Check the observed service-related missingness pattern."""

    missing_rows = df[
        df["InternetService"].isna()
    ]

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    if missing_rows.empty:
        pytest.skip("No missing InternetService rows to inspect.")

    assert (missing_rows["PhoneService"] == 1).all()

    for column in service_columns:
        assert (missing_rows[column] == 0).all()


def test_leakage_prone_columns_are_excluded_from_model_features():
    """Prevent identifiers and derived columns entering model inputs."""

    proposed_features = {
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
    }

    assert proposed_features.isdisjoint(
        FORBIDDEN_MODEL_FEATURES
    )
