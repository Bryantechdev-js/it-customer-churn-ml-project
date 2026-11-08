
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from churn_prediction.preprocessing import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    load_dataset,
    build_preprocessor,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

REPORTS_DIR = PROJECT_ROOT / "reports"

RESULTS_PATH = (
    REPORTS_DIR / "baseline_model_comparison.csv"
)


# ==================================================
# 2. LOGISTIC REGRESSION PREPROCESSOR
# ==================================================

def build_logistic_preprocessor():
    """
    Build preprocessing specifically for Logistic Regression.

    Numerical features are imputed and standardized.
    Categorical features are imputed and one-hot encoded.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
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

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ==================================================
# 3. BUILD MODEL PIPELINES
# ==================================================

def build_models():
    """
    Create complete preprocessing + classifier pipelines.

    Each model learns preprocessing only from the data
    passed to its fit() method.
    """

    models = {
        "DummyClassifier": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    DummyClassifier(
                        strategy="most_frequent"
                    ),
                ),
            ]
        ),

        "LogisticRegression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_logistic_preprocessor(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "RandomForestClassifier": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


# ==================================================
# 4. EVALUATE MODEL
# ==================================================

def evaluate_model(
    model,
    X_valid,
    y_valid,
    model_name,
):
    """
    Calculate classification and probability metrics.
    """

    predictions = model.predict(X_valid)

    probabilities = model.predict_proba(X_valid)[:, 1]

    metrics = {
        "model": model_name,

        "accuracy": accuracy_score(
            y_valid,
            predictions,
        ),

        "precision": precision_score(
            y_valid,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_valid,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_valid,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_valid,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_valid,
            probabilities,
        ),

        "brier_score": brier_score_loss(
            y_valid,
            probabilities,
        ),
    }

    matrix = confusion_matrix(
        y_valid,
        predictions,
        labels=[0, 1],
    )

    return metrics, matrix


# ==================================================
# 5. TRAIN AND COMPARE MODELS
# ==================================================

def run_model_comparison():

    print("\nLoading dataset...")

    df = load_dataset()

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN].astype(int)

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    models = build_models()

    results = []

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n========== MODEL TRAINING ==========")

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        # Fit the entire pipeline using training data.
        model.fit(X_train, y_train)

        # Evaluate on untouched validation data.
        metrics, matrix = evaluate_model(
            model,
            X_valid,
            y_valid,
            model_name,
        )

        results.append(metrics)

        print("\nMetrics:")

        for metric_name, value in metrics.items():

            if metric_name == "model":
                print(f"{metric_name}: {value}")
            else:
                print(f"{metric_name}: {value:.4f}")

        print("\nConfusion matrix:")

        print(matrix)

        print(
            "\nConfusion matrix interpretation:"
        )

        print(
            f"True negatives: {matrix[0, 0]}"
        )

        print(
            f"False positives: {matrix[0, 1]}"
        )

        print(
            f"False negatives: {matrix[1, 0]}"
        )

        print(
            f"True positives: {matrix[1, 1]}"
        )

    # Create comparison report.
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")

    print(
        results_df.round(4).to_string(index=False)
    )

    print(
        f"\nComparison report saved to: {RESULTS_PATH}"
    )


# ==================================================
# 6. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    run_model_comparison()
