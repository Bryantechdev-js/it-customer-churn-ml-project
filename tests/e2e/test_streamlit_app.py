"""Local end-to-end smoke checks for the Streamlit customer journey."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from churn_prediction.preprocessing import FEATURE_COLUMNS, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "app.py"


def test_no_internet_and_no_phone_customer_can_be_assessed():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception

    app.radio(key="navigation").set_value("Predict Customer").run(timeout=30)
    app.selectbox(key="phone_service").set_value("No")
    app.selectbox(key="internet_service").set_value(
        "No internet service"
    )
    app.button(key="predict_customer_submit").click().run(timeout=30)

    assert not app.exception
    assert app.session_state["last_customer_prediction"]
    assert app.session_state["last_customer_features"]["MultipleLines"] == (
        "No phone service"
    )
    assert app.session_state["last_customer_features"]["OnlineSecurity"] == (
        "No internet service"
    )


def test_out_of_training_range_input_is_flagged_in_result():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio(key="navigation").set_value("Predict Customer").run(timeout=30)
    app.number_input(key="tenure_input").set_value(100)
    app.button(key="predict_customer_submit").click().run(timeout=30)

    assert not app.exception
    assert app.session_state["last_customer_prediction"][
        "out_of_training_range"
    ] is True


def test_all_main_pages_render_without_streamlit_exceptions():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception

    for page in [
        "Predict Customer",
        "Batch Analysis",
        "Insights & Reports",
        "Model Health",
        "Overview",
    ]:
        app.radio(key="navigation").set_value(page).run(timeout=30)
        assert not app.exception, f"Page failed to render: {page}"


def test_batch_upload_and_prediction_flow():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio(key="navigation").set_value("Batch Analysis").run(timeout=30)

    sample = load_dataset().loc[:, FEATURE_COLUMNS].head(2)
    csv_bytes = sample.to_csv(index=False).encode("utf-8")
    app.file_uploader(key="batch_csv_uploader").upload(
        "sample.csv",
        csv_bytes,
        "text/csv",
    ).run(timeout=30)

    assert not app.exception
    app.button(key="run_batch_prediction").click().run(timeout=30)

    assert not app.exception
    assert len(app.session_state["last_batch_result"]) == 2
    assert app.session_state["last_batch_result"][
        "churn_probability"
    ].between(0, 1).all()
