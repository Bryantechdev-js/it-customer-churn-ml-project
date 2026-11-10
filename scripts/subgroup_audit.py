"""Generate descriptive holdout metrics by existing audit-only attributes."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_prediction.governance import subgroup_performance_report
from churn_prediction.inference import load_model_bundle, predict_batch
from churn_prediction.model_export import prepare_train_test_data


def main():
    dataset, _, X_test, _, y_test = prepare_train_test_data()
    _, metadata = load_model_bundle()
    predictions = predict_batch(X_test)

    groups = {
        "gender": dataset.loc[X_test.index, "gender"].to_numpy(),
        "SeniorCitizen": dataset.loc[
            X_test.index,
            "SeniorCitizen",
        ].to_numpy(),
    }

    report = subgroup_performance_report(
        y_true=y_test.to_numpy(),
        probabilities=predictions["churn_probability"].to_numpy(),
        group_columns=groups,
        threshold=float(metadata["threshold"]),
    )

    output_path = PROJECT_ROOT / "reports" / "subgroup_performance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print(report.round(4).to_string(index=False))
    print(f"\nDiagnostic report saved to: {output_path}")
    print(
        "This descriptive report does not establish fairness, "
        "statistical significance, or legal compliance."
    )


if __name__ == "__main__":
    main()
