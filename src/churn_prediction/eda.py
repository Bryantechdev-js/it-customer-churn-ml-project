
"""Exploratory Data Analysis for IT Customer Churn Intelligence."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

# Suitable for automated execution without opening chart windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "it_customer_churn.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "Churn"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


# ---------------------------------------------------------
# 2. DIRECTORY AND DATA LOADING
# ---------------------------------------------------------

def create_output_directories():
    """Create directories required for EDA outputs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load the raw dataset and validate essential columns."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    invalid_targets = set(df[TARGET].dropna().unique()) - {0, 1}

    if invalid_targets:
        raise ValueError(
            f"Unexpected target values: {invalid_targets}"
        )

    return df


# ---------------------------------------------------------
# 3. TARGET VARIABLE ANALYSIS
# ---------------------------------------------------------

def analyze_target(df):
    """Analyze churn distribution and class imbalance."""

    counts = df[TARGET].value_counts().sort_index()
    percentages = (
        df[TARGET].value_counts(normalize=True).sort_index() * 100
    )

    results = {
        "total_customers": int(len(df)),
        "class_counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "class_percentages": {
            str(key): round(float(value), 4)
            for key, value in percentages.items()
        },
    }

    with open(
        REPORT_DIR / "target_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=TARGET,
        hue=TARGET,
        palette="Set2",
        legend=False,
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = Retained, 1 = Churned)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "churn_distribution.png",
        dpi=150,
    )
    plt.close()

    return results


# ---------------------------------------------------------
# 4. NUMERICAL FEATURE ANALYSIS
# ---------------------------------------------------------

def analyze_numeric_features(df):
    """Calculate numerical statistics and create histograms."""

    available_columns = [
        col for col in NUMERIC_COLUMNS if col in df.columns
    ]

    numeric_df = df[available_columns].copy()

    for column in available_columns:
        numeric_df[column] = pd.to_numeric(
            numeric_df[column],
            errors="coerce",
        )

    summary = numeric_df.describe().T

    summary["missing_values"] = numeric_df.isna().sum()

    summary.to_csv(
        REPORT_DIR / "numeric_summary.csv"
    )

    for column in available_columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=numeric_df,
            x=column,
            kde=True,
            bins=30,
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{column}_distribution.png",
            dpi=150,
        )

        plt.close()

    return summary


# ---------------------------------------------------------
# 5. CATEGORICAL FEATURE ANALYSIS
# ---------------------------------------------------------

def analyze_categorical_features(df):
    """Calculate customer counts and churn rates per category."""

    results = {}

    for column in CATEGORICAL_COLUMNS:

        if column not in df.columns:
            continue

        grouped = (
            df.groupby(column, dropna=False)[TARGET]
            .agg(
                customers="count",
                churned="sum",
                churn_rate="mean",
            )
            .reset_index()
        )

        grouped["churn_rate_percent"] = (
            grouped["churn_rate"] * 100
        ).round(2)

        grouped["churn_rate"] = grouped["churn_rate"].round(4)

        results[column] = grouped.to_dict(orient="records")

        safe_name = column.lower().replace(" ", "_")

        grouped.to_csv(
            REPORT_DIR / f"churn_by_{safe_name}.csv",
            index=False,
        )

        # Plot churn rate by category.
        plot_data = grouped.copy()

        plot_data[column] = plot_data[column].fillna(
            "Missing"
        ).astype(str)

        plt.figure(figsize=(10, 5))

        sns.barplot(
            data=plot_data,
            x=column,
            y="churn_rate_percent",
            hue=column,
            palette="viridis",
            legend=False,
        )

        plt.title(f"Churn Rate by {column}")
        plt.xlabel(column)
        plt.ylabel("Churn Rate (%)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"churn_rate_by_{safe_name}.png",
            dpi=150,
        )

        plt.close()

    with open(
        REPORT_DIR / "categorical_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4, default=str)

    return results


# ---------------------------------------------------------
# 6. MISSING VALUE ANALYSIS
# ---------------------------------------------------------

def analyze_missing_values(df):
    """Inspect missingness and investigate InternetService."""

    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
    )

    missing["missing_percentage"] = (
        missing["missing_count"] / len(df) * 100
    ).round(2)

    missing = missing[missing["missing_count"] > 0]

    missing.to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    investigation = {}

    if "InternetService" in df.columns:

        related_columns = [
            "PhoneService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        available = [
            col for col in related_columns if col in df.columns
        ]

        missing_internet = df[
            df["InternetService"].isna()
        ]

        investigation = {
            "missing_internet_service_rows": int(
                len(missing_internet)
            ),
            "related_feature_distributions": {},
        }

        for column in available:

            counts = (
                missing_internet[column]
                .value_counts(dropna=False)
                .to_dict()
            )

            investigation["related_feature_distributions"][
                column
            ] = {
                str(key): int(value)
                for key, value in counts.items()
            }

    with open(
        REPORT_DIR / "missing_value_investigation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(investigation, file, indent=4)

    return missing, investigation


# ---------------------------------------------------------
# 7. FEATURE CORRELATION
# ---------------------------------------------------------

def analyze_correlations(df):
    """Inspect correlations between numeric features."""

    numeric_df = df[
        [
            col
            for col in NUMERIC_COLUMNS + [TARGET]
            if col in df.columns
        ]
    ].copy()

    for column in numeric_df.columns:
        numeric_df[column] = pd.to_numeric(
            numeric_df[column],
            errors="coerce",
        )

    correlation = numeric_df.corr(numeric_only=True)

    correlation.to_csv(
        REPORT_DIR / "numeric_correlations.csv"
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
    )

    plt.title("Numeric Feature Correlation")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "numeric_correlation_heatmap.png",
        dpi=150,
    )

    plt.close()

    return correlation


# ---------------------------------------------------------
# 8. MAIN EDA PIPELINE
# ---------------------------------------------------------

def run_eda():

    create_output_directories()

    print("\nLoading dataset...")
    df = load_dataset()

    print(f"Dataset shape: {df.shape}")

    print("\nAnalyzing target variable...")
    target_results = analyze_target(df)

    print("\nAnalyzing numerical features...")
    numeric_results = analyze_numeric_features(df)

    print("\nAnalyzing categorical features...")
    categorical_results = analyze_categorical_features(df)

    print("\nAnalyzing missing values...")
    missing_results, internet_investigation = (
        analyze_missing_values(df)
    )

    print("\nAnalyzing correlations...")
    correlation_results = analyze_correlations(df)

    print("\nEDA COMPLETED")
    print("------------------------------")
    print(f"Customers analyzed: {len(df)}")
    print(f"Features analyzed: {len(df.columns)}")
    print(
        "Churn distribution:",
        target_results["class_counts"],
    )
    print(
        "Missing-value columns:",
        missing_results.index.tolist(),
    )
    print(
        "Reports directory: {REPORT_DIR}".format(
            REPORT_DIR=REPORT_DIR
        )
    )
    print(
        "Figures directory: {FIGURE_DIR}".format(
            FIGURE_DIR=FIGURE_DIR
        )
    )


if __name__ == "__main__":
    run_eda()
