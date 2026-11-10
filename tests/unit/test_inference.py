"""Inference contracts and regression coverage for the exported model."""

import numpy as np
import pandas as pd
import pytest

from churn_prediction.inference import (
    FEATURE_COLUMNS,
    INTERNET_ADDON_FEATURES,
    MAX_BATCH_ROWS,
    ArtifactError,
    InputValidationError,
    load_model_bundle,
    normalize_ui_customer_features,
    predict_batch,
    predict_customer,
    validate_feature_frame,
)


@pytest.fixture(scope="module")
def model_metadata():
    return load_model_bundle()[1]


def make_ui_customer(**overrides):
    customer = {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
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
    customer.update(overrides)
    return customer


def frame_from_ui(customer):
    normalized = normalize_ui_customer_features(customer)
    return pd.DataFrame([normalized], columns=FEATURE_COLUMNS)


def test_model_bundle_matches_runtime_and_checksum(model_metadata):
    assert model_metadata["model_name"] == "LogisticRegression"
    assert model_metadata["scikit_learn_version"]
    assert model_metadata["model_artifact_sha256"]


@pytest.mark.parametrize("phone_service", ["Yes", "No"])
def test_no_internet_service_is_valid_with_or_without_phone(
    model_metadata,
    phone_service,
):
    customer = make_ui_customer(
        PhoneService=phone_service,
        MultipleLines=(
            "No phone service" if phone_service == "No" else "No"
        ),
        InternetService="No internet service",
        **{feature: "No internet service" for feature in INTERNET_ADDON_FEATURES},
    )

    validated = validate_feature_frame(
        frame_from_ui(customer),
        model_metadata,
    )

    assert pd.isna(validated.loc[0, "InternetService"])
    assert validated.loc[0, "PhoneService"] == int(phone_service == "Yes")
    assert (validated.loc[0, INTERNET_ADDON_FEATURES] == 0).all()


def test_legacy_not_recorded_ui_label_remains_supported(model_metadata):
    customer = make_ui_customer(
        InternetService="Not recorded",
        **{feature: "No internet service" for feature in INTERNET_ADDON_FEATURES},
    )
    validated = validate_feature_frame(frame_from_ui(customer), model_metadata)
    assert pd.isna(validated.loc[0, "InternetService"])


def test_no_internet_text_in_batch_maps_to_missing_category(model_metadata):
    customer = make_ui_customer(
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="No internet service",
        **{feature: "No internet service" for feature in INTERNET_ADDON_FEATURES},
    )
    normalized = normalize_ui_customer_features(customer)
    batch = pd.DataFrame([normalized])
    batch["InternetService"] = "No internet service"

    validated = validate_feature_frame(batch, model_metadata)

    assert pd.isna(validated.loc[0, "InternetService"])
    assert validated.loc[0, "PhoneService"] == 0


def test_duplicate_batch_columns_are_rejected(model_metadata):
    batch = frame_from_ui(make_ui_customer())
    duplicate = pd.concat([batch, batch[["Partner"]]], axis=1)

    with pytest.raises(InputValidationError, match="duplicate column"):
        validate_feature_frame(duplicate, model_metadata)


def test_ui_input_must_be_a_mapping():
    with pytest.raises(InputValidationError, match="mapping"):
        normalize_ui_customer_features(None)


def test_no_internet_with_active_internet_addon_is_rejected(model_metadata):
    customer = make_ui_customer(
        InternetService="No internet service",
        OnlineSecurity="Yes",
        **{
            feature: "No internet service"
            for feature in INTERNET_ADDON_FEATURES
            if feature != "OnlineSecurity"
        },
    )

    with pytest.raises(InputValidationError, match="internet add-on"):
        validate_feature_frame(frame_from_ui(customer), model_metadata)


def test_no_phone_with_multiple_lines_is_rejected(model_metadata):
    customer = make_ui_customer(PhoneService="No", MultipleLines="Yes")

    with pytest.raises(InputValidationError, match="without phone service"):
        validate_feature_frame(frame_from_ui(customer), model_metadata)


@pytest.mark.parametrize(
    ("feature", "value"),
    [
        ("tenure", -1),
        ("tenure", 1.5),
        ("MonthlyCharges", np.inf),
        ("TotalCharges", np.nan),
        ("Partner", 2),
        ("Contract", "Lifetime"),
        ("PaymentMethod", "Cash"),
        ("InternetService", "Satellite"),
    ],
)
def test_invalid_feature_values_are_rejected(model_metadata, feature, value):
    batch = frame_from_ui(make_ui_customer()).astype(object)
    batch.loc[0, feature] = value

    with pytest.raises(InputValidationError):
        validate_feature_frame(batch, model_metadata)


def test_single_prediction_is_finite_and_in_range():
    result = predict_customer(make_ui_customer())

    assert np.isfinite(result["churn_probability"])
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["predicted_class"] in (0, 1)
    assert result["threshold_status"]
    expected_assessment = (
        "Churn predicted"
        if result["predicted_class"] == 1
        else "No churn predicted"
    )
    assert result["risk_assessment"] == expected_assessment


def test_single_prediction_flags_values_outside_development_ranges():
    result = predict_customer(make_ui_customer(tenure=100))

    assert result["out_of_training_range"] is True
    assert result["out_of_range_features"] == ["tenure"]


def test_batch_prediction_preserves_extras_and_matches_single_prediction():
    customer = make_ui_customer()
    normalized = normalize_ui_customer_features(customer)
    batch = pd.DataFrame([{**normalized, "external_reference": "case-001"}])
    original = batch.copy(deep=True)

    result = predict_batch(batch)
    single = predict_customer(customer)

    assert batch.equals(original)
    assert result.loc[0, "external_reference"] == "case-001"
    assert result.loc[0, "predicted_class"] == single["predicted_class"]
    assert result.loc[0, "churn_probability"] == pytest.approx(
        single["churn_probability"]
    )


def test_batch_out_of_range_warning_is_per_record():
    normalized = normalize_ui_customer_features(make_ui_customer())
    in_range = dict(normalized)
    out_of_range = {**normalized, "tenure": 100}
    batch = pd.DataFrame([in_range, out_of_range])

    result = predict_batch(batch)

    assert result["out_of_training_range"].tolist() == [False, True]
    assert result["out_of_range_features"].tolist() == ["", "tenure"]


def test_batch_rejects_output_column_collisions():
    normalized = normalize_ui_customer_features(make_ui_customer())
    batch = pd.DataFrame([{**normalized, "source_row": "upstream"}])

    with pytest.raises(InputValidationError, match="reserved output"):
        predict_batch(batch)


def test_batch_enforces_row_limit_before_loading_model():
    batch = pd.DataFrame(
        {"placeholder": range(MAX_BATCH_ROWS + 1)}
    )

    with pytest.raises(InputValidationError, match="maximum"):
        predict_batch(batch)


def test_invalid_artifact_metadata_fails_closed(tmp_path):
    from churn_prediction import inference

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("[]", encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"not loaded")

    with pytest.raises(ArtifactError, match="JSON object"):
        inference._load_bundle_cached(
            str(model_path),
            str(metadata_path),
            model_path.stat().st_mtime_ns,
            metadata_path.stat().st_mtime_ns,
        )


def test_artifact_checksum_mismatch_fails_before_deserialization(tmp_path):
    import json

    from churn_prediction import inference

    _, source_metadata = load_model_bundle()
    metadata = dict(source_metadata)
    metadata["model_artifact_sha256"] = "0" * 64
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"untrusted artifact content")

    with pytest.raises(ArtifactError, match="checksum"):
        inference._load_bundle_cached(
            str(model_path),
            str(metadata_path),
            model_path.stat().st_mtime_ns,
            metadata_path.stat().st_mtime_ns,
        )
