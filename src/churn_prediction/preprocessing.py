
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "it_customer_churn.csv"
)

TARGET_COLUMN = "Churn"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# Numerical features
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


# Categorical features
CATEGORICAL_FEATURES = [
    "Partner",
    "Dependents",
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
]


FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the raw dataset and validate required columns,
    target values, and basic data integrity.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    target_values = set(
        df[TARGET_COLUMN].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Target must contain only binary values 0 and 1."
        )

    if len(target_values) != 2:
        raise ValueError(
            "Target must contain both classes: 0 and 1."
        )

    # Validate numerical feature types.
    for column in NUMERIC_FEATURES:

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Numerical feature '{column}' "
                "contains a non-numeric data type."
            )

        if np.isinf(
            df[column].dropna().to_numpy()
        ).any():
            raise ValueError(
                f"Feature '{column}' contains infinite values."
            )

    return df


# ==================================================
# 3. BUILD NUMERICAL PIPELINE
# ==================================================

def build_numeric_pipeline(
    scale: bool = False,
) -> Pipeline:
    """
    Handle missing numerical values using the median.

    Optionally standardize numerical features.

    Scaling is useful for models such as Logistic Regression
    because numerical features can have very different scales.

    The imputer and scaler learn their parameters only
    from the data passed to fit().
    """

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(
        steps=steps
    )


# ==================================================
# 4. BUILD CATEGORICAL PIPELINE
# ==================================================

def build_categorical_pipeline() -> Pipeline:
    """
    Handle missing categorical values and encode
    categorical variables into numerical features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


# ==================================================
# 5. BUILD COLUMN TRANSFORMER
# ==================================================

def build_preprocessor(
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Apply different preprocessing pipelines to
    numerical and categorical features.

    scale_numeric=False:
        Impute numerical features without scaling.

    scale_numeric=True:
        Impute and scale numerical features.

    Categorical features are imputed and one-hot encoded
    without applying numerical scaling to their indicators.
    """

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                build_numeric_pipeline(
                    scale=scale_numeric
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                build_categorical_pipeline(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ==================================================
# 6. VALIDATE TRANSFORMED FEATURES
# ==================================================

def validate_processed_data(
    X_train_processed,
    X_valid_processed,
    preprocessor,
) -> None:
    """
    Validate transformed feature dimensions,
    generated feature names, and missing values.
    """

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # Handle sparse and dense matrices.
    if sparse.issparse(X_train_processed):
        processed_values = X_train_processed.data
    else:
        processed_values = X_train_processed

    if sparse.issparse(X_valid_processed):
        validation_values = X_valid_processed.data
    else:
        validation_values = X_valid_processed

    contains_train_nan = np.isnan(
        processed_values
    ).any()

    contains_valid_nan = np.isnan(
        validation_values
    ).any()

    # Verify dimensions.
    assert (
        len(feature_names)
        == X_train_processed.shape[1]
    ), "Feature name count does not match training dimensions."

    assert (
        X_train_processed.shape[1]
        == X_valid_processed.shape[1]
    ), "Training and validation feature counts differ."

    assert not contains_train_nan, (
        "Training features contain NaN values."
    )

    assert not contains_valid_nan, (
        "Validation features contain NaN values."
    )

    print("\n========== FEATURE VALIDATION ==========")

    print(
        f"Number of generated features: "
        f"{len(feature_names)}"
    )

    print("\nFirst 15 generated features:")

    for feature in feature_names[:15]:
        print(f"- {feature}")

    print(
        f"\nTraining contains NaN: "
        f"{contains_train_nan}"
    )

    print(
        f"Validation contains NaN: "
        f"{contains_valid_nan}"
    )

    print(
        f"Training matrix shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Validation matrix shape: "
        f"{X_valid_processed.shape}"
    )

    print("\nFeature validation passed.")


# ==================================================
# 7. SPLIT DATA AND RUN PREPROCESSING
# ==================================================

def run_preprocessing() -> None:
    """
    Load data, split it, fit preprocessing on training
    data, and transform validation data.
    """

    print("\nLoading dataset...")

    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    print("Dataset loaded successfully.")

    # Split before fitting preprocessing.
    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Default: do not scale numerical features.
    preprocessor = build_preprocessor(
        scale_numeric=False
    )

    # Fit transformations ONLY on training data.
    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    # Apply learned transformations to validation data.
    X_valid_processed = (
        preprocessor.transform(X_valid)
    )

    # Validate generated features.
    validate_processed_data(
        X_train_processed,
        X_valid_processed,
        preprocessor,
    )

    # ==================================================
    # REPORT
    # ==================================================

    print("\n========== PREPROCESSING REPORT ==========")

    print(f"Original dataset: {X.shape}")

    print(f"Training data: {X_train.shape}")

    print(f"Validation data: {X_valid.shape}")

    print("\nTarget distribution:")

    print("\nTraining:")

    print(
        y_train.value_counts(
            normalize=True
        ).sort_index().round(4)
    )

    print("\nValidation:")

    print(
        y_valid.value_counts(
            normalize=True
        ).sort_index().round(4)
    )

    print("\nOriginal missing values in training data:")

    missing_values = (
        X_train.isna().sum()
    )

    print(
        missing_values[
            missing_values > 0
        ]
    )

    print("\nProcessed feature shapes:")

    print(
        f"Training: {X_train_processed.shape}"
    )

    print(
        f"Validation: {X_valid_processed.shape}"
    )

    print("\nPreprocessing completed successfully.")


# ==================================================
# 8. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    run_preprocessing()
