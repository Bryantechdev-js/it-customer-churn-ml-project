
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from churn_prediction.preprocessing import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    load_dataset,
    build_preprocessor,
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

REPORT_DIR = PROJECT_ROOT / "reports"


# ==================================================
# 2. PREPARE DEVELOPMENT DATA
# ==================================================

def prepare_development_data():
    """
    Load the dataset and separate development data
    from the final test set.

    The final test set remains untouched during
    model selection and cross-validation.
    """

    print("\nLoading dataset...")

    df = load_dataset()

    X = df[FEATURE_COLUMNS]
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

    print("\nDataset split completed.")

    print(f"Development rows: {len(X_development)}")
    print(f"Final test rows: {len(X_final_test)}")

    print("\nDevelopment target distribution:")

    print(
        y_development.value_counts(
            normalize=True
        ).sort_index().round(4)
    )

    # Do not evaluate or inspect final test performance here.
    return X_development, y_development


# ==================================================
# 3. BUILD CROSS-VALIDATOR
# ==================================================

def build_cross_validator():
    """
    Create stratified five-fold cross-validation.

    Each validation fold maintains approximately
    the same target class distribution.
    """

    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ==================================================
# 4. BUILD MODELS
# ==================================================

def build_models():
    """
    Define candidate classification algorithms.

    Logistic Regression:
        Increased iterations to support convergence.

    Random Forest:
        Restricted tree complexity to reduce overfitting.
    """

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
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
# 5. BUILD MODEL PIPELINE
# ==================================================

def build_model_pipeline(model):
    """
    Build a leakage-safe preprocessing and model pipeline.

    Logistic Regression:
        Scale numerical features only.

    Random Forest:
        Do not scale numerical features.

    Categorical features are imputed and one-hot encoded
    inside the preprocessing pipeline.

    During cross-validation, preprocessing is fitted
    independently on each training fold.
    """

    if isinstance(model, LogisticRegression):
        preprocessor = build_preprocessor(
            scale_numeric=True
        )

    else:
        preprocessor = build_preprocessor(
            scale_numeric=False
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


# ==================================================
# 6. DEFINE EVALUATION METRICS
# ==================================================

def get_scoring_metrics():
    """
    Return evaluation metrics for binary classification.

    Brier score is returned as negative Brier score
    by sklearn because cross_validate maximizes scores.

    We convert it back to its positive form later.
    """

    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "neg_brier_score": "neg_brier_score",
    }


# ==================================================
# 7. EVALUATE MODELS
# ==================================================

def evaluate_models(X, y, cv, models):
    """
    Evaluate candidate models using stratified
    cross-validation.

    Both training and validation metrics are retained
    to help identify overfitting.
    """

    scoring = get_scoring_metrics()

    fold_results = []

    print("\n========== 5-FOLD CROSS-VALIDATION ==========")

    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

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

        for fold_index in range(N_SPLITS):

            fold_result = {
                "model": model_name,
                "fold": fold_index + 1,

                # Validation metrics
                "accuracy": scores["test_accuracy"][fold_index],
                "precision": scores["test_precision"][fold_index],
                "recall": scores["test_recall"][fold_index],
                "f1": scores["test_f1"][fold_index],
                "roc_auc": scores["test_roc_auc"][fold_index],
                "pr_auc": scores["test_pr_auc"][fold_index],

                # Training metrics
                "train_accuracy": scores["train_accuracy"][fold_index],
                "train_precision": scores["train_precision"][fold_index],
                "train_recall": scores["train_recall"][fold_index],
                "train_f1": scores["train_f1"][fold_index],
                "train_roc_auc": scores["train_roc_auc"][fold_index],
                "train_pr_auc": scores["train_pr_auc"][fold_index],

                # Positive Brier scores
                "brier_score": -scores["test_neg_brier_score"][fold_index],
                "train_brier_score": -scores["train_neg_brier_score"][fold_index],
            }

            fold_results.append(fold_result)

    return pd.DataFrame(fold_results)


# ==================================================
# 8. SUMMARIZE CROSS-VALIDATION RESULTS
# ==================================================

def summarize_results(fold_results):
    """
    Calculate mean and standard deviation
    for every validation metric.
    """

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

    for model_name in fold_results["model"].unique():

        model_data = fold_results[
            fold_results["model"] == model_name
        ]

        summary = {
            "model": model_name,
        }

        for metric in metrics:

            summary[f"{metric}_mean"] = (
                model_data[metric].mean()
            )

            summary[f"{metric}_std"] = (
                model_data[metric].std()
            )

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    # Higher PR-AUC is preferred.
    summary_df = summary_df.sort_values(
        by="pr_auc_mean",
        ascending=False,
    )

    return summary_df.reset_index(drop=True)


# ==================================================
# 9. SAVE REPORTS
# ==================================================

def save_reports(fold_results, summary_results):
    """
    Save fold-level and summary-level reports.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_report_path = (
        REPORT_DIR / "cv_fold_results.csv"
    )

    summary_report_path = (
        REPORT_DIR / "cv_model_comparison.csv"
    )

    fold_results.to_csv(
        fold_report_path,
        index=False,
    )

    summary_results.to_csv(
        summary_report_path,
        index=False,
    )

    print("\nReports saved:")

    print(fold_report_path)

    print(summary_report_path)


# ==================================================
# 10. PRINT RESULTS
# ==================================================

def print_results(fold_results, summary_results):
    """
    Display individual fold results and
    aggregate cross-validation results.
    """

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        200,
    )

    pd.set_option(
        "display.float_format",
        lambda value: f"{value:.4f}",
    )

    print("\n========== INDIVIDUAL FOLD RESULTS ==========")

    print(
        fold_results.to_string(
            index=False
        )
    )

    print("\n========== CROSS-VALIDATION SUMMARY ==========")

    print(
        summary_results.to_string(
            index=False
        )
    )


# ==================================================
# 11. RUN EXPERIMENT
# ==================================================

def run_experiment():
    """
    Execute the complete cross-validation experiment.
    """

    X, y = prepare_development_data()

    cv = build_cross_validator()

    models = build_models()

    fold_results = evaluate_models(
        X=X,
        y=y,
        cv=cv,
        models=models,
    )

    summary_results = summarize_results(
        fold_results
    )

    print_results(
        fold_results,
        summary_results,
    )

    save_reports(
        fold_results,
        summary_results,
    )

    print("\nCross-validation completed successfully.")


# ==================================================
# 12. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    run_experiment()
