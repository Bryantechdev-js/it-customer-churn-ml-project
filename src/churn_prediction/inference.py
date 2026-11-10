
"""
IT Customer Churn Intelligence
Inference service.

Responsibilities:
- Load the trusted local model artifact.
- Validate artifact metadata and model schema.
- Convert UI labels into the model's original encoding.
- Validate single-customer and batch inputs.
- Generate churn probabilities and provisional classifications.

Never load model files supplied by users.
"""

from functools import lru_cache
from collections.abc import Mapping
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline

from churn_prediction.preprocessing import (
    PROJECT_ROOT,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "churn_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
MAX_BATCH_ROWS = 10_000

BINARY_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

INTERNET_ADDON_FEATURES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# ============================================================
# 2. EXCEPTIONS
# ============================================================

class InferenceError(Exception):
    """Base exception for inference-related errors."""


class ArtifactError(InferenceError):
    """Raised when the model artifact is missing or invalid."""


class InputValidationError(InferenceError):
    """Raised when customer input does not match the model schema."""


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# 3. TRUSTED ARTIFACT LOADING
# ============================================================

@lru_cache(maxsize=4)
def _load_bundle_cached(
    model_path_string,
    metadata_path_string,
    model_modified_ns,
    metadata_modified_ns,
):
    """
    Load and validate a locally trusted model artifact.

    The modification timestamps are part of the cache key,
    so replacing the artifact invalidates the previous cache.
    """

    model_path = Path(model_path_string)
    metadata_path = Path(metadata_path_string)

    if not model_path.is_file():
        raise ArtifactError(
            f"Model artifact not found: {model_path}"
        )

    if not metadata_path.is_file():
        raise ArtifactError(
            f"Model metadata not found: {metadata_path}"
        )

    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"Unable to read model metadata: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ArtifactError("Model metadata must be a JSON object.")

    required_metadata_fields = [
        "model_name",
        "model_version",
        "feature_columns",
        "numeric_features",
        "categorical_features",
        "model_classes",
        "threshold",
        "categorical_values",
        "numeric_feature_ranges",
        "python_version",
        "scikit_learn_version",
        "model_artifact_sha256",
    ]

    missing_metadata_fields = [
        field
        for field in required_metadata_fields
        if field not in metadata
    ]

    if missing_metadata_fields:
        raise ArtifactError(
            "Model metadata is incomplete. Missing: "
            + ", ".join(missing_metadata_fields)
        )

    recorded_sklearn = metadata.get("scikit_learn_version")
    if recorded_sklearn and recorded_sklearn != sklearn.__version__:
        raise ArtifactError(
            "The model was exported with scikit-learn "
            f"{recorded_sklearn}, but this runtime has "
            f"{sklearn.__version__}. Install the pinned requirements "
            "and redeploy."
        )

    recorded_python = metadata.get("python_version")
    if recorded_python:
        recorded_minor = ".".join(str(recorded_python).split(".")[:2])
        runtime_minor = ".".join(platform.python_version().split(".")[:2])
        if recorded_minor != runtime_minor:
            raise ArtifactError(
                "The model was exported with Python "
                f"{recorded_minor}, but this runtime has {runtime_minor}. "
                "Redeploy with the model's Python minor version."
            )

    expected_artifact_hash = metadata.get("model_artifact_sha256")
    if expected_artifact_hash:
        actual_artifact_hash = _sha256_file(model_path)
        if actual_artifact_hash.lower() != str(expected_artifact_hash).lower():
            raise ArtifactError(
                "The model artifact checksum does not match its metadata."
            )

    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ArtifactError(
            "The artifact feature schema does not match "
            "the application's approved feature schema."
        )

    if metadata["numeric_features"] != NUMERIC_FEATURES:
        raise ArtifactError(
            "Numeric feature schema mismatch."
        )

    if metadata["categorical_features"] != CATEGORICAL_FEATURES:
        raise ArtifactError(
            "Categorical feature schema mismatch."
        )

    categorical_values = metadata["categorical_values"]
    if not isinstance(categorical_values, dict):
        raise ArtifactError("Categorical value metadata must be an object.")
    missing_category_metadata = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature not in categorical_values
        or not isinstance(categorical_values[feature], list)
    ]
    if missing_category_metadata:
        raise ArtifactError(
            "Categorical value metadata is incomplete for: "
            + ", ".join(missing_category_metadata)
        )

    numeric_feature_ranges = metadata["numeric_feature_ranges"]
    if (
        not isinstance(numeric_feature_ranges, dict)
        or set(numeric_feature_ranges) != set(NUMERIC_FEATURES)
    ):
        raise ArtifactError("Numeric feature range metadata must be an object.")
    for feature in NUMERIC_FEATURES:
        feature_range = numeric_feature_ranges.get(feature)
        if (
            not isinstance(feature_range, dict)
            or not {"min", "max"}.issubset(feature_range)
        ):
            raise ArtifactError(
                f"Numeric range metadata is incomplete for {feature}."
            )
        try:
            lower = float(feature_range["min"])
            upper = float(feature_range["max"])
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"Numeric range metadata is invalid for {feature}."
            ) from exc
        if (
            not np.isfinite([lower, upper]).all()
            or lower < 0
            or lower > upper
        ):
            raise ArtifactError(
                f"Numeric range metadata is invalid for {feature}."
            )

    try:
        threshold = float(metadata["threshold"])

    except (TypeError, ValueError) as exc:
        raise ArtifactError(
            "The model threshold is invalid."
        ) from exc

    if not 0.0 <= threshold <= 1.0:
        raise ArtifactError(
            "The model threshold must be between 0 and 1."
        )

    # IMPORTANT:
    # joblib uses Python object deserialization.
    # Only load artifacts created and controlled by this project.
    try:
        model = joblib.load(model_path)

    except Exception as exc:
        raise ArtifactError(
            f"Unable to load the trusted model artifact: {exc}"
        ) from exc

    if not isinstance(model, Pipeline):
        raise ArtifactError(
            "The saved model is not a scikit-learn Pipeline."
        )

    if not {"preprocessor", "classifier"}.issubset(
        set(model.named_steps)
    ):
        raise ArtifactError(
            "The pipeline must contain preprocessor and classifier steps."
        )

    if not hasattr(model, "feature_names_in_"):
        raise ArtifactError(
            "The model does not expose its original input feature names."
        )

    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        raise ArtifactError(
            "The fitted model feature order does not match "
            "the approved feature schema."
        )

    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "classes_"):
        raise ArtifactError(
            "The classifier has no fitted class information."
        )

    actual_classes = [
        int(value)
        for value in classifier.classes_
    ]

    if actual_classes != [0, 1]:
        raise ArtifactError(
            f"Unexpected classifier classes: {actual_classes}"
        )

    if metadata["model_classes"] != actual_classes:
        raise ArtifactError(
            "Classifier classes do not match artifact metadata."
        )

    if not hasattr(model, "predict_proba"):
        raise ArtifactError(
            "The model does not support probability predictions."
        )

    return model, metadata


def load_model_bundle():
    """
    Public artifact-loading function.

    Only the configured project artifact paths are used.
    """

    if not MODEL_PATH.is_file():
        raise ArtifactError(
            f"Model artifact is missing: {MODEL_PATH}"
        )

    if not METADATA_PATH.is_file():
        raise ArtifactError(
            f"Model metadata is missing: {METADATA_PATH}"
        )

    return _load_bundle_cached(
        str(MODEL_PATH.resolve()),
        str(METADATA_PATH.resolve()),
        MODEL_PATH.stat().st_mtime_ns,
        METADATA_PATH.stat().st_mtime_ns,
    )


# ============================================================
# 4. UI LABEL NORMALIZATION
# ============================================================

def _map_yes_no(value, feature_name):
    """Convert a UI Yes/No selection to the training encoding."""

    mapping = {
        "Yes": 1,
        "No": 0,
    }

    if value not in mapping:
        raise InputValidationError(
            f"{feature_name}: expected Yes or No; "
            f"received {value!r}."
        )

    return mapping[value]


def _normalize_internet_service_value(value):
    """Map known no-internet labels to the model's missing-value category."""

    if pd.isna(value):
        return np.nan

    normalized = str(value).strip()
    if normalized.casefold() in {
        "",
        "no",
        "no internet service",
        "not recorded",
    }:
        return np.nan

    return normalized


def _out_of_training_range(data, metadata):
    """Return a row flag and feature names outside development data ranges."""

    flags = np.zeros(len(data), dtype=bool)
    feature_names = [[] for _ in range(len(data))]

    for feature, feature_range in metadata["numeric_feature_ranges"].items():
        values = data[feature]
        outside = (
            (values < float(feature_range["min"]))
            | (values > float(feature_range["max"]))
        )
        outside_positions = np.flatnonzero(outside.to_numpy())
        flags[outside_positions] = True
        for position in outside_positions:
            feature_names[position].append(feature)

    return (
        pd.Series(flags, index=data.index, dtype=bool),
        pd.Series(
            [";".join(features) for features in feature_names],
            index=data.index,
            dtype=object,
        ),
    )


def normalize_ui_customer_features(ui_features):
    """
    Translate the existing Streamlit form labels into
    the original 17-feature model schema.

    This function does not make predictions.
    """

    if not isinstance(ui_features, Mapping):
        raise InputValidationError(
            "Customer input must be a mapping of feature names to values."
        )

    required_ui_fields = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
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

    missing_fields = [
        field
        for field in required_ui_fields
        if field not in ui_features
    ]

    if missing_fields:
        raise InputValidationError(
            "The customer form is missing fields: "
            + ", ".join(missing_fields)
        )

    normalized = {
        "tenure": ui_features["tenure"],
        "MonthlyCharges": ui_features["MonthlyCharges"],
        "TotalCharges": ui_features["TotalCharges"],
    }

    for feature in [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]:
        normalized[feature] = _map_yes_no(
            ui_features[feature],
            feature,
        )

    normalized["MultipleLines"] = {
        "Yes": 1,
        "No": 0,
        "No phone service": 0,
    }.get(
        ui_features["MultipleLines"],
        None,
    )

    if normalized["MultipleLines"] is None:
        raise InputValidationError(
            "MultipleLines has an unsupported selection."
        )

    for feature in INTERNET_ADDON_FEATURES:
        ui_value = ui_features[feature]

        mapping = {
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
        }

        if ui_value not in mapping:
            raise InputValidationError(
                f"{feature} has an unsupported selection: "
                f"{ui_value!r}."
            )

        normalized[feature] = mapping[ui_value]

    internet_mapping = {
        "DSL": "DSL",
        "Fiber optic": "Fiber-optic",
        # The training data has a missing-value category here. Map the
        # explicit UI choice to that learned category. Keep the old UI label
        # as a compatibility alias for already-integrated callers.
        "No internet service": np.nan,
        "Not recorded": np.nan,
    }

    internet_value = ui_features["InternetService"]

    if internet_value not in internet_mapping:
        raise InputValidationError(
            "InternetService has an unsupported selection."
        )

    normalized["InternetService"] = (
        internet_mapping[internet_value]
    )

    contract_mapping = {
        "Month-to-month": "Month-to-month",
        "One year": "One-year",
        "Two year": "Two-year",
    }

    contract_value = ui_features["Contract"]

    if contract_value not in contract_mapping:
        raise InputValidationError(
            "Contract has an unsupported selection."
        )

    normalized["Contract"] = contract_mapping[contract_value]

    payment_mapping = {
        "Electronic check": "Electronic-check",
        "Mailed check": "Mailed-check",
        "Bank transfer (automatic)": "Bank-transfer",
        "Credit card (automatic)": "Credit-card",
    }

    payment_value = ui_features["PaymentMethod"]

    if payment_value not in payment_mapping:
        raise InputValidationError(
            "PaymentMethod has an unsupported selection."
        )

    normalized["PaymentMethod"] = payment_mapping[payment_value]

    return normalized


# ============================================================
# 5. FEATURE VALIDATION
# ============================================================

def validate_feature_frame(data, metadata):
    """
    Validate and return a DataFrame in the exact model schema.

    Additional columns are ignored by the model.
    All 17 approved features are mandatory.
    """

    if not isinstance(data, pd.DataFrame):
        raise InputValidationError(
            "Input must be a pandas DataFrame."
        )

    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise InputValidationError(
            "Input contains duplicate column names: "
            + ", ".join(map(str, duplicates))
        )

    if data.empty:
        raise InputValidationError(
            "No customer records were provided."
        )

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise InputValidationError(
            "Missing required model features: "
            + ", ".join(missing_columns)
        )

    result = data.loc[:, FEATURE_COLUMNS].copy()

    # --------------------------------------------------------
    # Numeric feature validation
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        values = result[column].to_numpy(dtype=float)

        invalid_mask = (
            ~np.isfinite(values)
            | (values < 0)
        )

        if invalid_mask.any():
            bad_rows = (
                np.where(invalid_mask)[0] + 1
            ).tolist()

            raise InputValidationError(
                f"{column} contains missing, non-finite or "
                f"negative values at input rows: {bad_rows[:10]}."
            )

    # Tenure represents whole months.
    if not np.equal(
        result["tenure"],
        np.floor(result["tenure"]),
    ).all():
        raise InputValidationError(
            "Tenure must contain whole numbers of months."
        )

    result["tenure"] = result["tenure"].astype(int)

    # --------------------------------------------------------
    # Binary feature validation
    # --------------------------------------------------------

    for column in BINARY_FEATURES:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        invalid_mask = (
            result[column].isna()
            | ~result[column].isin([0, 1])
        )

        if invalid_mask.any():
            bad_rows = (
                np.where(invalid_mask)[0] + 1
            ).tolist()

            raise InputValidationError(
                f"{column} must contain only 0 or 1. "
                f"Invalid input rows: {bad_rows[:10]}."
            )

        result[column] = result[column].astype(int)

    # --------------------------------------------------------
    # InternetService normalization
    # --------------------------------------------------------

    result["InternetService"] = result[
        "InternetService"
    ].map(_normalize_internet_service_value)

    # --------------------------------------------------------
    # Other categorical feature validation
    # --------------------------------------------------------

    string_categorical_features = [
        "Contract",
        "PaymentMethod",
    ]

    allowed_categories = metadata["categorical_values"]

    for column in string_categorical_features:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
        )

        if result[column].isna().any():
            raise InputValidationError(
                f"{column} cannot contain missing values."
            )

        allowed = set(
            allowed_categories[column]
        )

        invalid_values = sorted(
            set(result[column].astype(str)) - allowed
        )

        if invalid_values:
            raise InputValidationError(
                f"{column} contains unsupported categories: "
                f"{invalid_values}."
            )

        result[column] = result[column].astype(object)

    # InternetService categories are validated separately
    # because missing values are allowed.
    allowed_internet = set(
        allowed_categories["InternetService"]
    )

    actual_internet = set(
        result["InternetService"].dropna().astype(str)
    )

    invalid_internet = actual_internet - allowed_internet

    if invalid_internet:
        raise InputValidationError(
            "InternetService contains unsupported categories: "
            f"{sorted(invalid_internet)}."
        )

    # --------------------------------------------------------
    # Cross-feature consistency checks
    # --------------------------------------------------------

    invalid_phone_rows = (
        (result["PhoneService"] == 0)
        & (result["MultipleLines"] != 0)
    )

    if invalid_phone_rows.any():
        raise InputValidationError(
            "A customer without phone service cannot have "
            "MultipleLines = 1."
        )

    no_internet_mask = result["InternetService"].isna()

    if no_internet_mask.any():
        invalid_no_internet = (
            result.loc[
                no_internet_mask,
                INTERNET_ADDON_FEATURES,
            ].sum(axis=1) != 0
        )

        if invalid_no_internet.any():
            raise InputValidationError(
                "When InternetService is 'No internet service', "
                "all internet add-on features must be No. "
                "Phone service is independent and may be Yes or No."
            )

    return result


# ============================================================
# 6. POSITIVE CLASS PROBABILITY
# ============================================================

def _positive_class_probabilities(model, validated_data):
    """Extract probabilities for the class representing churn."""

    classifier = model.named_steps["classifier"]

    classes = np.asarray(classifier.classes_)

    positive_indices = np.where(classes == 1)[0]

    if len(positive_indices) != 1:
        raise ArtifactError(
            "The model does not have exactly one positive class."
        )

    probabilities = model.predict_proba(validated_data)

    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(validated_data)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ArtifactError(
            "The model returned invalid class probabilities."
        )

    return probabilities[:, positive_indices[0]]


# ============================================================
# 7. PREDICTION
# ============================================================

def predict_customer(ui_features):
    """
    Generate a single-customer churn risk estimate.

    ui_features uses the labels from the existing Streamlit form.
    """

    model, metadata = load_model_bundle()

    normalized = normalize_ui_customer_features(
        ui_features
    )

    customer_frame = pd.DataFrame(
        [normalized],
        columns=FEATURE_COLUMNS,
    )

    validated = validate_feature_frame(
        customer_frame,
        metadata,
    )

    probability = float(
        _positive_class_probabilities(
            model,
            validated,
        )[0]
    )
    out_of_range, out_of_range_features = _out_of_training_range(
        validated,
        metadata,
    )

    threshold = float(metadata["threshold"])

    predicted_class = int(
        probability >= threshold
    )

    return {
        "churn_probability": probability,
        "predicted_class": predicted_class,
        "risk_assessment": (
            "Above provisional threshold"
            if predicted_class == 1
            else "Below provisional threshold"
        ),
        "threshold": threshold,
        "threshold_status": metadata["threshold_status"],
        "out_of_training_range": bool(out_of_range.iloc[0]),
        "out_of_range_features": (
            out_of_range_features.iloc[0].split(";")
            if bool(out_of_range.iloc[0])
            else []
        ),
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
    }


def predict_batch(data):
    """
    Generate predictions for a batch DataFrame.

    The input must contain all 17 model features.
    Additional input columns are preserved in the output,
    but are never passed to the model.
    """

    if not isinstance(data, pd.DataFrame):
        raise InputValidationError(
            "Input must be a pandas DataFrame."
        )

    if data.empty:
        raise InputValidationError(
            "No customer records were provided."
        )

    if len(data) > MAX_BATCH_ROWS:
        raise InputValidationError(
            f"Batch exceeds the maximum of {MAX_BATCH_ROWS:,} rows."
        )

    reserved_output_columns = {
        "source_row",
        "churn_probability",
        "predicted_class",
        "risk_assessment",
        "threshold",
        "threshold_status",
        "out_of_training_range",
        "out_of_range_features",
    }
    collisions = sorted(reserved_output_columns.intersection(data.columns))
    if collisions:
        raise InputValidationError(
            "Input contains reserved output columns: "
            + ", ".join(collisions)
        )

    model, metadata = load_model_bundle()

    validated = validate_feature_frame(
        data,
        metadata,
    )

    probabilities = _positive_class_probabilities(
        model,
        validated,
    )

    threshold = float(metadata["threshold"])

    results = data.copy()

    results.insert(
        0,
        "source_row",
        np.arange(1, len(results) + 1),
    )

    results["churn_probability"] = probabilities

    out_of_range, out_of_range_features = _out_of_training_range(
        validated,
        metadata,
    )
    results["out_of_training_range"] = out_of_range.to_numpy()
    results["out_of_range_features"] = out_of_range_features.to_numpy()

    results["predicted_class"] = (
        probabilities >= threshold
    ).astype(int)

    results["risk_assessment"] = np.where(
        results["predicted_class"] == 1,
        "Above provisional threshold",
        "Below provisional threshold",
    )

    results["threshold"] = threshold

    results["threshold_status"] = (
        metadata["threshold_status"]
    )

    return results
