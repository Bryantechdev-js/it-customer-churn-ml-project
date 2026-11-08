
"""
IT Customer Churn Intelligence
Out-of-fold probability and threshold analysis.

The reserved final test set is not used here.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import cross_val_predict

from churn_prediction.preprocessing import PROJECT_ROOT
from churn_prediction.cv_model_evaluation import (
    prepare_development_data,
    build_cross_validator,
    build_models,
    build_model_pipeline,
)


REPORT_DIR = PROJECT_ROOT / "reports"

THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70,
]


def evaluate_thresholds(y_true, probabilities, model_name):
    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        results.append({
            "model": model_name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, predictions
            ),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "specificity": specificity,
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "customers_flagged": int(predictions.sum()),
            "flagged_percentage": predictions.mean() * 100,
        })

    return pd.DataFrame(results)


def run_threshold_analysis():
    print("\nLoading development data...")

    X, y = prepare_development_data()
    cv = build_cross_validator()
    models = build_models()

    all_results = []

    for model_name, model in models.items():
        print(f"\nGenerating out-of-fold probabilities: {model_name}")

        pipeline = build_model_pipeline(model)

        probabilities = cross_val_predict(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        print("ROC-AUC:", round(roc_auc_score(y, probabilities), 4))
        print(
            "PR-AUC:",
            round(average_precision_score(y, probabilities), 4),
        )
        print(
            "Brier score:",
            round(brier_score_loss(y, probabilities), 4),
        )

        results = evaluate_thresholds(
            y_true=y,
            probabilities=probabilities,
            model_name=model_name,
        )

        all_results.append(results)

    results_df = pd.concat(all_results, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "threshold_analysis.csv"
    results_df.to_csv(output_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n========== THRESHOLD ANALYSIS ==========")
    print(results_df.round(4).to_string(index=False))

    print(f"\nReport saved to: {output_path}")
    print("\nThreshold analysis completed successfully.")


if __name__ == "__main__":
    run_threshold_analysis()
