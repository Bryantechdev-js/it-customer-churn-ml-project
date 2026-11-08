
"""
IT Customer Churn Intelligence
Five-Fold Stratified Cross-Validation

Objectives:
1. Compare Logistic Regression and Random Forest.
2. Prevent preprocessing leakage.
3. Scale features for Logistic Regression.
4. Investigate Random Forest overfitting.
5. Evaluate models across five folds.
6. Save fold-level and summary reports.

The final test set remains untouched.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from churn_prediction.preprocessing import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    build_preprocessor,
    load_dataset,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# 2. LOAD AND SPLIT DATA
# ==================================================

def prepare_development_data():

    print("\nLoading dataset...")

    df = load_dataset()

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].astype(int).copy()

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print("\nDataset split completed.")

    print(f"Development rows: {len(X_development)}")

    print(f"Final test rows: {len(X_test)}")

    print("\nDevelopment target distribution:")

    print(
        y_development
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    # Important:
    # The final test set is deliberately not used
    # during cross-validation or model selection.

    return X_development, y_development


# ==================================================
# 3. DEFINE CROSS-VALIDATION
# ==================================================

def build_cross_validator():

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ==================================================
# 4. DEFINE MODELS
# ==================================================

def build_models():

    models = {

        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),

        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),

    }

    return models


# ==================================================
# 5. BUILD PREPROCESSING + MODEL PIPELINE
# ==================================================

def build_model_pipeline(model):

    steps = [
        ("preprocessor", build_preprocessor()),
    ]

    # Logistic Regression benefits from feature scaling.
    #
    # with_mean=False preserves compatibility with
    # sparse matrices produced by OneHotEncoder.
    #
    # Random Forest does not require feature scaling.

    if isinstance(model, LogisticRegression):

        steps.append(
            (
                "scaler",
                StandardScaler(with_mean=False),
            )
        )

    steps.append(
        ("classifier", model)
    )

    pipeline = Pipeline(
        steps=steps
    )

    return pipeline


# ==================================================
# 6. DEFINE EVALUATION METRICS
# ==================================================

def get_scoring_metrics():

    scoring = {

        "accuracy": "accuracy",

        "precision": "precision",

        "recall": "recall",

        "f1": "f1",

        "roc_auc": "roc_auc",

        "pr_auc": "average_precision",

        "neg_brier_score": "neg_brier_score",

    }

    return scoring


# ==================================================
# 7. EVALUATE MODELS USING CROSS-VALIDATION
# ==================================================

def evaluate_models(X, y):

    cv = build_cross_validator()

    models = build_models()

    scoring = get_scoring_metrics()

    all_fold_results = []

    print(
        "\n========== 5-FOLD CROSS-VALIDATION =========="
    )

    for model_name, model in models.items():

        print(
            f"\nEvaluating {model_name}..."
        )

        pipeline = build_model_pipeline(model)

        scores = cross_validate(

            estimator=pipeline,

            X=X,

            y=y,

            cv=cv,

            scoring=scoring,

            return_train_score=True,

            n_jobs=-1,

            error_score="raise",

        )

        # Collect results for each fold.

        for fold_index in range(N_SPLITS):

            fold_result = {

                "model": model_name,

                "fold": fold_index + 1,

            }

            metrics = [

                "accuracy",

                "precision",

                "recall",

                "f1",

                "roc_auc",

                "pr_auc",

            ]

            for metric in metrics:

                fold_result[metric] = scores[
                    f"test_{metric}"
                ][fold_index]

                fold_result[
                    f"train_{metric}"
                ] = scores[
                    f"train_{metric}"
                ][fold_index]

            # sklearn returns negative Brier scores
            # because cross_validate assumes higher
            # scores are better.

            fold_result["brier_score"] = -scores[
                "test_neg_brier_score"
            ][fold_index]

            fold_result["train_brier_score"] = -scores[
                "train_neg_brier_score"
            ][fold_index]

            all_fold_results.append(
                fold_result
            )

    return pd.DataFrame(all_fold_results)


# ==================================================
# 8. SUMMARIZE CROSS-VALIDATION RESULTS
# ==================================================

def summarize_results(results):

    metrics = [

        "accuracy",

        "precision",

        "recall",

        "f1",

        "roc_auc",

        "pr_auc",

        "brier_score",

    ]

    summary_rows = []

    for model_name, group in results.groupby("model"):

        row = {
            "model": model_name
        }

        for metric in metrics:

            row[f"{metric}_mean"] = (
                group[metric].mean()
            )

            row[f"{metric}_std"] = (
                group[metric].std()
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    # Higher PR-AUC is preferred for this
    # imbalanced churn classification problem.

    summary = summary.sort_values(
        by="pr_auc_mean",
        ascending=False,
    )

    return summary


# ==================================================
# 9. SAVE REPORTS
# ==================================================

def save_reports(fold_results, summary):

    fold_path = (
        REPORT_DIR / "cv_fold_results.csv"
    )

    summary_path = (
        REPORT_DIR / "cv_model_comparison.csv"
    )

    fold_results.to_csv(
        fold_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n========== REPORTS SAVED ==========")

    print(f"Fold results: {fold_path}")

    print(f"Model comparison: {summary_path}")


# ==================================================
# 10. RUN COMPLETE EXPERIMENT
# ==================================================

def run_experiment():

    # Prepare development data.

    X_development, y_development = (
        prepare_development_data()
    )

    # Evaluate models.

    fold_results = evaluate_models(
        X_development,
        y_development,
    )

    # Calculate mean and standard deviation.

    summary = summarize_results(
        fold_results
    )

    # Display fold-level results.

    print(
        "\n========== INDIVIDUAL FOLD RESULTS =========="
    )

    print(
        fold_results.round(4).to_string(
            index=False
        )
    )

    # Display summary.

    print(
        "\n========== CROSS-VALIDATION SUMMARY =========="
    )

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    # Save reports.

    save_reports(
        fold_results,
        summary,
    )

    print(
        "\nCross-validation completed successfully."
    )


# ==================================================
# 11. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_experiment()
