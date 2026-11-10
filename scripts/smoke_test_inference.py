import sys
import traceback
from numbers import Real
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from churn_prediction.inference import (
    load_model_bundle,
    predict_customer,
    predict_batch,
    normalize_ui_customer_features,
    InputValidationError,
    FEATURE_COLUMNS,
)


# ============================================================
# IT CUSTOMER CHURN INTELLIGENCE
# INFERENCE SERVICE SMOKE TEST
# ============================================================

passed_tests = []
failed_tests = []

model = None
metadata = None
customer = None
normalized = None
batch_input = None


def record_pass(name):
    passed_tests.append(name)
    print(f"[PASS] {name}")


def record_failure(name, error):
    failed_tests.append((name, str(error)))
    print(f"[FAIL] {name}: {error}")


def run_test(name, function):
    try:
        function()
    except Exception as exc:
        record_failure(name, exc)
        traceback.print_exc()
    else:
        record_pass(name)


# ------------------------------------------------------------
# TEST 1: MODEL AND METADATA LOADING
# ------------------------------------------------------------

def test_model_loading():
    global model, metadata

    model, metadata = load_model_bundle()

    assert model is not None, "Model is None."
    assert metadata is not None, "Metadata is None."

    assert "model_name" in metadata
    assert "model_version" in metadata
    assert "threshold" in metadata
    assert "threshold_status" in metadata

    assert 0.0 <= float(metadata["threshold"]) <= 1.0


# ------------------------------------------------------------
# TEST 2: INPUT NORMALIZATION
# ------------------------------------------------------------

def test_input_normalization():
    global customer, normalized

    customer = {
        "tenure": 12,
        "MonthlyCharges": 70.00,
        "TotalCharges": 840.00,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    normalized = normalize_ui_customer_features(customer)

    # Validate membership, not dictionary insertion order.
    assert set(normalized.keys()) == set(FEATURE_COLUMNS), (
        "Normalized features do not match the required feature schema."
    )

    assert len(normalized) == len(FEATURE_COLUMNS)


# ------------------------------------------------------------
# TEST 3: SINGLE CUSTOMER PREDICTION
# ------------------------------------------------------------

def test_single_prediction():
    assert customer is not None, "Customer fixture is unavailable."

    result = predict_customer(customer)

    probability = result["churn_probability"]

    assert isinstance(probability, Real), (
        "Churn probability must be a numeric value."
    )

    probability = float(probability)

    assert 0.0 <= probability <= 1.0

    assert result["predicted_class"] in (0, 1)

    assert "risk_assessment" in result
    assert "threshold" in result
    assert "model_name" in result
    assert "model_version" in result

    print(f"       Churn probability: {probability:.2%}")
    print(f"       Predicted class: {result['predicted_class']}")
    print(f"       Risk assessment: {result['risk_assessment']}")


# ------------------------------------------------------------
# TEST 4: BATCH PREDICTION
# ------------------------------------------------------------

def test_batch_prediction():
    global batch_input

    assert normalized is not None, "Normalized customer is unavailable."

    batch_input = pd.DataFrame([
        {
            **normalized,
            "test_reference": "CUSTOMER-001",
        },
        {
            **normalized,
            "tenure": 48,
            "MonthlyCharges": 55.00,
            "TotalCharges": 2640.00,
            "Contract": "Two-year",
            "test_reference": "CUSTOMER-002",
        },
    ])

    result = predict_batch(batch_input)

    assert len(result) == 2

    required_output_columns = [
        "source_row",
        "churn_probability",
        "predicted_class",
        "risk_assessment",
        "threshold",
        "threshold_status",
        "test_reference",
    ]

    for column in required_output_columns:
        assert column in result.columns, (
            f"Missing batch output column: {column}"
        )

    assert result["churn_probability"].between(0, 1).all()

    assert result["predicted_class"].isin([0, 1]).all()

    print("\n       Batch prediction results:")

    print(
        result[
            [
                "source_row",
                "test_reference",
                "churn_probability",
                "predicted_class",
                "risk_assessment",
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# TEST 5: INVALID CUSTOMER INPUT REJECTION
# ------------------------------------------------------------

def test_invalid_customer_rejection():
    invalid_customer = customer.copy()

    invalid_customer["Contract"] = "Lifetime contract"

    try:
        predict_customer(invalid_customer)
    except InputValidationError:
        return

    raise AssertionError(
        "Invalid contract was incorrectly accepted."
    )


# ------------------------------------------------------------
# TEST 6: MISSING BATCH FEATURE REJECTION
# ------------------------------------------------------------

def test_missing_batch_feature_rejection():
    incomplete_batch = batch_input.drop(
        columns=["MonthlyCharges"]
    )

    try:
        predict_batch(incomplete_batch)
    except InputValidationError:
        return

    raise AssertionError(
        "Batch with a missing required feature was accepted."
    )


# ============================================================
# EXECUTE TESTS
# ============================================================

print("\n" + "=" * 65)
print("IT CUSTOMER CHURN INTELLIGENCE")
print("INFERENCE SERVICE SMOKE TEST")
print("=" * 65)

tests = [
    ("Model and metadata loading", test_model_loading),
    ("UI input normalization", test_input_normalization),
    ("Single customer prediction", test_single_prediction),
    ("Batch customer prediction", test_batch_prediction),
    ("Invalid customer input rejection", test_invalid_customer_rejection),
    ("Missing batch feature rejection", test_missing_batch_feature_rejection),
]

for test_name, test_function in tests:
    print(f"\n[TEST] {test_name}")
    run_test(test_name, test_function)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("SMOKE TEST SUMMARY")
print("=" * 65)

print(f"Passed: {len(passed_tests)}")
print(f"Failed: {len(failed_tests)}")
print(f"Total:  {len(tests)}")

if metadata is not None:
    print("\nMODEL INFORMATION")
    print("Model:", metadata.get("model_name"))
    print("Version:", metadata.get("model_version"))
    print("Threshold:", metadata.get("threshold"))
    print("Threshold status:", metadata.get("threshold_status"))

if failed_tests:
    print("\nFAILED TESTS:")

    for name, error in failed_tests:
        print(f"- {name}: {error}")

    print("\nSMOKE TEST FAILED.")
    print("Resolve the reported issues before UI integration.")

    sys.exit(1)

print("\nALL SMOKE TESTS PASSED.")

print("\nIMPORTANT:")
print("These tests use synthetic customer records.")
print("Predictions are estimates, not verified customer outcomes.")
print("A passing smoke test does not establish model accuracy.")

sys.exit(0)
