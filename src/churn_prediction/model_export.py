
"""
IT Customer Churn Intelligence
Reproducible model training, final evaluation and artifact export.

The final test set is reserved for this evaluation.
It is not used to fit the model or select a threshold.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from churn_prediction.preprocessing import (
    PROJECT_ROOT,
    DATA_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    load_dataset,
)
from churn_prediction.cv_model_evaluation import (
    build_models,
    build_model_pipeline,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Provisional candidate selected using development CV F1/recall.
MODEL_NAME = "LogisticRegression"

# This is a technical starting point, NOT an approved
# operational/business threshold.
PROVISIONAL_THRESHOLD = 0.50

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "churn_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


# ============================================================
# 2. DATA FINGERPRINT
# ============================================================

def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 3. PREPARE DEVELOPMENT AND FINAL TEST DATA
# ============================================================

def prepare_train_test_data():
    """
    Reproduce the same stratified split used during
    cross-validation.

    Only development data is used to fit the pipeline.
    The final test set is reserved for final evaluation.
    """

    df = load_dataset()

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    X_development, X_final_test, y_development, y_final_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    return (
        df,
        X_development,
        X_final_test,
        y_development,
        y_final_test,
    )


# ============================================================
# 4. POSITIVE CLASS PROBABILITY
# ============================================================

def get_positive_class_probabilities(model, X):
    """
    Find the probability column corresponding to class 1.

    Never assume that predict_proba() column 1 always
    represents the positive class.
    """

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The selected model does not support predict_proba()."
        )

    classifier = model.named_steps["classifier"]

    classes = np.asarray(classifier.classes_)

    positive_indices = np.where(classes == 1)[0]

    if len(positive_indices) != 1:
        raise ValueError(
            f"Expected exactly one positive class (1). "
            f"Found classes: {classes.tolist()}"
        )

    probabilities = model.predict_proba(X)

    return probabilities[:, positive_indices[0]]


# ============================================================
# 5. FINAL TEST EVALUATION
# ============================================================

def evaluate_final_test(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the fitted pipeline on the reserved final test set.

    The threshold is provisional and is not optimized here.
    """

    probabilities = get_positive_class_probabilities(
        model,
        X_test,
    )

    predictions = (
        probabilities >= PROVISIONAL_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_test,
                probabilities,
            )
        ),
        "threshold": PROVISIONAL_THRESHOLD,
        "threshold_status": "provisional_not_business_approved",
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "test_rows": int(len(X_test)),
    }

    return metrics


# ============================================================
# 6. BUILD MODEL METADATA
# ============================================================

def build_metadata(
    model,
    model_name,
    development_data,
    final_metrics,
):
    """Create traceable metadata for the exported model."""

    category_values = {}

    for column in CATEGORICAL_FEATURES:
        category_values[column] = sorted(
            development_data[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    classifier = model.named_steps["classifier"]

    metadata = {
        "model_name": model_name,
        "model_version": "1.0.0",
        "artifact_filename": MODEL_PATH.name,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "target_column": TARGET_COLUMN,

        "target_mapping": {
            "0": "retained",
            "1": "churned",
        },

        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ],
        "categorical_features": CATEGORICAL_FEATURES,

        "categorical_values": category_values,

        "model_classes": [
            int(value)
            for value in classifier.classes_
        ],

        "training_rows": int(len(development_data)),

        "training_data_sha256": calculate_sha256(
            DATA_PATH
        ),

        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,

        "selection_basis": (
            "Provisional development choice based on "
            "cross-validation F1 and recall."
        ),

        "threshold": PROVISIONAL_THRESHOLD,
        "threshold_status": (
            "Provisional technical default; "
            "not approved as an operational policy."
        ),

        "final_test_metrics": final_metrics,

        "limitations": [
            "Predictions are estimates, not guarantees.",
            "Probability calibration has not been independently approved.",
            "The operating threshold requires business validation.",
            "Final test metrics describe this held-out dataset only.",
        ],
    }

    return metadata


# ============================================================
# 7. EXPORT ARTIFACTS
# ============================================================

def export_artifacts(model, metadata):
    """Save the trusted pipeline and metadata."""

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_model_path = (
        ARTIFACT_DIR / "churn_pipeline.tmp.joblib"
    )

    temporary_metadata_path = (
        ARTIFACT_DIR / "model_metadata.tmp.json"
    )

    # Save complete preprocessing + classifier pipeline.
    joblib.dump(
        model,
        temporary_model_path,
    )

    temporary_model_path.replace(MODEL_PATH)

    # Save metadata separately for inspection.
    with temporary_metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    temporary_metadata_path.replace(METADATA_PATH)


# ============================================================
# 8. MAIN TRAINING WORKFLOW
# ============================================================

def run_export():

    print("\n========== MODEL EXPORT WORKFLOW ==========")

    print(f"Selected candidate: {MODEL_NAME}")

    (
        df,
        X_development,
        X_final_test,
        y_development,
        y_final_test,
    ) = prepare_train_test_data()

    print("\n========== DATA SPLIT ==========")

    print(f"Full dataset: {len(df):,}")
    print(f"Development rows: {len(X_development):,}")
    print(f"Final test rows: {len(X_final_test):,}")

    models = build_models()

    if MODEL_NAME not in models:
        raise ValueError(
            f"Unknown model: {MODEL_NAME}"
        )

    pipeline = build_model_pipeline(
        models[MODEL_NAME]
    )

    print("\n========== FINAL MODEL TRAINING ==========")

    # Fit only on development data.
    pipeline.fit(
        X_development,
        y_development,
    )

    print("Model training completed.")

    print("\n========== FINAL TEST EVALUATION ==========")

    final_metrics = evaluate_final_test(
        pipeline,
        X_final_test,
        y_final_test,
    )

    for name, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\n========== EXPORT ==========")

    metadata = build_metadata(
        model=pipeline,
        model_name=MODEL_NAME,
        development_data=X_development,
        final_metrics=final_metrics,
    )

    export_artifacts(
        model=pipeline,
        metadata=metadata,
    )

    print(f"\nPipeline saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")

    print("\nModel export completed successfully.")

    print(
        "\nIMPORTANT: The final test set has been used for "
        "evaluation. Do not use it to repeatedly tune "
        "the model or threshold."
    )


if __name__ == "__main__":
    run_export()
