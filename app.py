
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st  # type: ignore[import-not-found]

# Make the src-layout project package importable when Streamlit is launched elsewhere.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from churn_prediction.inference import (
        MAX_BATCH_ROWS,
        load_model_bundle,
        predict_customer,
        predict_batch,
    )
    from churn_prediction.exporting import safe_csv_bytes
except ModuleNotFoundError as exc:
    # Streamlit Cloud displays this message in the app UI, so missing runtime
    # packages are actionable even when the platform redacts its traceback.
    missing_package = exc.name or "a required package"
    st.error(
        f"The inference environment is missing `{missing_package}`. "
        "Install the packages listed in the project `requirements.txt`, "
        "then redeploy the app."
    )
    st.stop()


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Churn Intelligence | Customer Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = ROOT / "data" / "raw" / "it_customer_churn.csv"
CV_REPORT_PATH = ROOT / "reports" / "cv_model_comparison.csv"
THRESHOLD_REPORT_PATH = ROOT / "reports" / "threshold_analysis.csv"
SUBGROUP_REPORT_PATH = ROOT / "reports" / "subgroup_performance.csv"

MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

INTERNET_ADDON_UI_FIELDS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


# ============================================================
# 2. DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #F5F7FA;
        color: #172B4D;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 250, 0.95);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #14243A;
        letter-spacing: -0.4px;
    }

    p, label {
        color: #52647A;
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E6EBF1;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand {
        padding: 8px 8px 24px 8px;
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        background: #E5F6F3;
        color: #066B61;
        border-radius: 11px;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .brand-name {
        font-size: 17px;
        font-weight: 750;
        color: #14243A;
    }

    .brand-caption {
        font-size: 11px;
        color: #52647A;
        margin-top: 5px;
        letter-spacing: 0.7px;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: 14px;
        font-weight: 550;
    }

    .eyebrow {
        font-size: 11px;
        font-weight: 750;
        color: #066B61;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .page-title {
        font-size: 31px;
        font-weight: 750;
        color: #14243A;
        letter-spacing: -1px;
        margin-bottom: 7px;
    }

    .page-description {
        color: #718096;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 17px;
        font-weight: 700;
        color: #172B4D;
        margin: 18px 0 4px 0;
    }

    .section-description {
        color: #52647A;
        font-size: 12px;
        margin-bottom: 14px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E7ECF2;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 7px rgba(20, 36, 58, 0.025);
    }

    [data-testid="stMetricLabel"] {
        color: #52647A;
        font-size: 12px;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #14243A;
        font-size: 27px;
        font-weight: 750;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #E7ECF2;
        border-radius: 12px;
        background: #FFFFFF;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 8px;
    }

    .stButton button,
    .stFormSubmitButton button {
        background: #066B61;
        color: white;
        border: 1px solid #066B61;
        border-radius: 8px;
        min-height: 42px;
        font-weight: 650;
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover {
        background: #055A52;
        border-color: #055A52;
        color: white;
    }

    .stButton button:focus-visible,
    .stFormSubmitButton button:focus-visible,
    [data-baseweb="select"]:focus-within,
    [data-baseweb="input"]:focus-within {
        outline: 3px solid #7A4E00;
        outline-offset: 2px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #E7ECF2;
        border-radius: 10px;
        overflow: hidden;
    }

    .result-score {
        color: #14243A;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .result-label {
        color: #52647A;
        font-size: 12px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .app-footer {
        border-top: 1px solid #E5EAF0;
        padding-top: 18px;
        margin-top: 40px;
        color: #52647A;
        font-size: 11px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA ACCESS
# ============================================================

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_cv_report():
    if not CV_REPORT_PATH.exists():
        return None

    return pd.read_csv(CV_REPORT_PATH)


@st.cache_data
def load_threshold_report():
    if not THRESHOLD_REPORT_PATH.exists():
        return None

    return pd.read_csv(THRESHOLD_REPORT_PATH)


@st.cache_data
def load_subgroup_report():
    if not SUBGROUP_REPORT_PATH.exists():
        return None

    return pd.read_csv(SUBGROUP_REPORT_PATH)


@st.cache_data
def load_model_metadata():
    if not METADATA_PATH.exists():
        return None

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Invalid model metadata format.")

    return metadata


def check_model_status():
    if not MODEL_PATH.is_file():
        return False, "Model artifact is missing."

    if not METADATA_PATH.is_file():
        return False, "Model metadata is missing."

    try:
        load_model_bundle()
        return True, "Model loaded successfully."

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def format_percentage(value):
    if value is None:
        return "Unavailable"

    try:
        return f"{float(value):.2%}"
    except (ValueError, TypeError):
        return "Unavailable"


# ============================================================
# 4. REUSABLE UI COMPONENTS
# ============================================================

def render_header(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <div class="page-title">{title}</div>
        <div class="page-description">{description}</div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<div class="section-description">{description}</div>',
            unsafe_allow_html=True,
        )


def render_empty_state(title, description):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(description)


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            IT CUSTOMER CHURN INTELLIGENCE
            &nbsp; · &nbsp;
            Decision-support platform
            &nbsp; · &nbsp;
            Predictions are estimates, not guarantees.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(result):
    probability = float(result["churn_probability"])
    predicted_class = int(result["predicted_class"])
    threshold = float(result["threshold"])

    with st.container(border=True):
        st.markdown("#### Customer assessment")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                '<div class="result-label">Model score</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="result-score">{probability:.2%}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("**Predicted outcome**")

            if predicted_class == 1:
                st.error("Churn predicted")
            else:
                st.success("No churn predicted")

        with col3:
            st.markdown("**Provisional threshold**")
            st.metric("Threshold", f"{threshold:.2f}")

        st.progress(max(0.0, min(1.0, probability)))

        st.caption(
            f"Assessment: {result['risk_assessment']}"
        )

        st.divider()

        st.caption(
            f"Model: {result.get('model_name', 'Unknown')} | "
            f"Version: {result.get('model_version', 'Unknown')}"
        )

        st.warning(
            "This is an estimated model score, not a guarantee. "
            "Probability calibration has not been independently approved, "
            "and the decision threshold is not yet approved for business operations."
        )

        if result.get("out_of_training_range"):
            st.warning(
                "The following numeric values are outside the range seen "
                "during model development: "
                + ", ".join(result.get("out_of_range_features", []))
                + ". Treat the score as unsupported for this record."
            )


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">◈</div>
            <div class="brand-name">Churn Intelligence</div>
            <div class="brand-caption">
                Customer analytics platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("WORKSPACE")

    pages = [
        "Overview",
        "Predict Customer",
        "Batch Analysis",
        "Insights & Reports",
        "Model Health",
    ]

    page_icons = {
        "Overview": "⌂",
        "Predict Customer": "◎",
        "Batch Analysis": "▤",
        "Insights & Reports": "◫",
        "Model Health": "◇",
    }

    selected_page = st.radio(
        "Navigation",
        pages,
        format_func=lambda page: (
            f"{page_icons[page]}   {page}"
        ),
        label_visibility="collapsed",
        key="navigation",
    )

    st.divider()

    st.caption("ENVIRONMENT")
    st.markdown("**Local inference workspace**")
    st.caption("Model artifact loaded from project")

    st.divider()
    st.caption("VERSION 1.0.0")


# ============================================================
# 6. OVERVIEW
# ============================================================

def show_overview():

    render_header(
        "BUSINESS OVERVIEW",
        "Customer intelligence",
        "Understand observed customer retention, churn patterns and the analytical workspace.",
    )

    df = load_dataset()

    if df is None:
        st.warning(
            "Source dataset not found. "
            "Check data/raw/it_customer_churn.csv."
        )
        return

    if "Churn" not in df.columns:
        st.error("The dataset is missing the Churn target column.")
        return

    total_customers = len(df)
    churn_count = int((df["Churn"] == 1).sum())
    retained_count = int((df["Churn"] == 0).sum())

    churn_rate = (
        churn_count / total_customers * 100
        if total_customers else 0
    )

    retention_rate = (
        retained_count / total_customers * 100
        if total_customers else 0
    )

    render_section(
        "Portfolio summary",
        "Observed statistics from the loaded customer dataset.",
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total customers", f"{total_customers:,}")
    col2.metric("Churned customers", f"{churn_count:,}")
    col3.metric("Observed churn rate", f"{churn_rate:.2f}%")
    col4.metric("Observed retention rate", f"{retention_rate:.2f}%")

    left, right = st.columns([1.2, 0.8])

    with left:
        render_section(
            "Customer outcome distribution",
            "Observed churn and retention labels.",
        )

        outcome = pd.DataFrame(
            {
                "Customer outcome": ["Retained", "Churned"],
                "Customers": [retained_count, churn_count],
            }
        ).set_index("Customer outcome")

        st.bar_chart(outcome, horizontal=True)

    with right:
        render_section(
            "Data quality",
            "Missing values in the loaded source dataset.",
        )

        st.metric(
            "Missing values",
            f"{int(df.isna().sum().sum()):,}",
        )

        missing_internet = (
            int(df["InternetService"].isna().sum())
            if "InternetService" in df.columns else 0
        )

        st.metric(
            "Missing InternetService",
            f"{missing_internet:,}",
        )

        st.caption(
            "Missing values are processed by the model's fitted preprocessing pipeline."
        )

    render_section("Exported inference model")

    try:
        metadata = load_model_metadata()
    except Exception as exc:
        metadata = None
        st.error(f"Could not load model metadata: {exc}")

    model_ok, model_message = check_model_status()

    if model_ok:
        st.success("Local inference model is available.")
    else:
        st.warning(model_message)

    if metadata:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Model",
            str(metadata.get("model_name", "Unknown")),
        )

        c2.metric(
            "Version",
            str(metadata.get("model_version", "Unknown")),
        )

        training_rows = metadata.get("training_rows")

        c3.metric(
            "Training records",
            f"{training_rows:,}" if isinstance(training_rows, int) else "Unknown",
        )

        metrics = metadata.get("final_test_metrics", {})

        if metrics:

            render_section(
                "Held-out test performance",
                "Evaluation on the reserved test dataset.",
            )

            metric_cols = st.columns(4)

            for column, (label, key) in zip(
                metric_cols,
                [
                    ("Accuracy", "accuracy"),
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1-score", "f1"),
                ],
            ):
                column.metric(
                    label,
                    format_percentage(metrics.get(key)),
                )

            metric_cols_2 = st.columns(3)

            for column, (label, key) in zip(
                metric_cols_2,
                [
                    ("ROC-AUC", "roc_auc"),
                    ("PR-AUC", "pr_auc"),
                    ("Brier score", "brier_score"),
                ],
            ):
                value = metrics.get(key)

                column.metric(
                    label,
                    format_percentage(value),
                )

    render_section(
        "Cross-validation experiment",
        "Development-set model comparison.",
    )

    cv_report = load_cv_report()

    if cv_report is not None:

        display_columns = [
            column
            for column in [
                "model",
                "accuracy_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "roc_auc_mean",
                "pr_auc_mean",
            ]
            if column in cv_report.columns
        ]

        st.dataframe(
            cv_report[display_columns]
            if display_columns else cv_report,
            use_container_width=True,
            hide_index=True,
        )

    else:
        render_empty_state(
            "Model comparison report unavailable",
            "Run the cross-validation experiment first.",
        )

    st.caption(
        "Historical dataset statistics are not live production predictions."
    )


# ============================================================
# 7. SINGLE CUSTOMER PREDICTION
# ============================================================

def show_prediction():

    render_header(
        "CUSTOMER ASSESSMENT",
        "Assess churn risk",
        "Enter customer account and service information to generate a model-based assessment.",
    )

    model_ok, model_message = check_model_status()

    if not model_ok:
        st.error(f"Prediction unavailable: {model_message}")
        return

    st.success("Local inference model is available.")

    with st.form("customer_prediction_form"):

        render_section(
            "01 — Account profile",
            "Customer relationship and contract information.",
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12,
                key="tenure_input",
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"],
            )

        with col2:

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"],
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

        with col3:

            paperless_billing = st.selectbox(
                "Paperless billing",
                ["Yes", "No"],
            )

            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.divider()

        render_section(
            "02 — Service configuration",
            "Telephone and internet services.",
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            phone_service = st.selectbox(
                "Phone service",
                ["Yes", "No"],
                key="phone_service",
            )

            multiple_lines = st.selectbox(
                "Multiple lines",
                ["Yes", "No", "No phone service"],
                key="multiple_lines",
            )

        with col2:

            internet_service = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No internet service"],
                help=(
                    "Mapped to the source data's missing InternetService "
                    "category; confirm that missing means no service for "
                    "your data source."
                ),
                key="internet_service",
            )

            online_security = st.selectbox(
                "Online security",
                ["Yes", "No", "No internet service"],
                key="online_security",
            )

        with col3:

            online_backup = st.selectbox(
                "Online backup",
                ["Yes", "No", "No internet service"],
                key="online_backup",
            )

            device_protection = st.selectbox(
                "Device protection",
                ["Yes", "No", "No internet service"],
                key="device_protection",
            )

        st.divider()

        render_section(
            "03 — Additional internet services",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            tech_support = st.selectbox(
                "Technical support",
                ["Yes", "No", "No internet service"],
                key="tech_support",
            )

        with col2:
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
                key="streaming_tv",
            )

        with col3:
            streaming_movies = st.selectbox(
                "Streaming movies",
                ["Yes", "No", "No internet service"],
                key="streaming_movies",
            )

        st.divider()

        render_section(
            "04 — Financial information",
            "Monthly and cumulative customer charges.",
        )

        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                value=65.0,
                step=1.0,
                format="%.2f",
                key="monthly_charges_input",
            )

        with col2:
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                value=780.0,
                step=1.0,
                format="%.2f",
                key="total_charges_input",
            )

        submitted = st.form_submit_button(
            "Assess customer risk",
            use_container_width=True,
            key="predict_customer_submit",
        )

    if submitted:
        customer_features = {
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,

            # Keep this exact string.
            # Do not convert it to None.
            "InternetService": internet_service,

            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }

        # Canonicalize unavailable services from the explicit parent choices.
        # This avoids asking users to manually set every dependent service to
        # an unavailable value. The adjustments are shown before the result.
        adjustments = []
        if phone_service == "No":
            customer_features["MultipleLines"] = "No phone service"
            adjustments.append("Multiple lines set to unavailable")

        if internet_service == "No internet service":
            for feature in INTERNET_ADDON_UI_FIELDS:
                customer_features[feature] = "No internet service"
            adjustments.append(
                "Internet add-ons set to unavailable"
            )

        if adjustments:
            st.info(
                "Applied service constraints: "
                + "; ".join(adjustments)
                + "."
            )

        st.session_state.pop(
            "last_customer_prediction",
            None,
        )

        st.session_state["last_customer_features"] = customer_features

        try:

            with st.spinner("Calculating customer assessment..."):

                result = predict_customer(customer_features)

            st.session_state["last_customer_prediction"] = result

            st.success("Customer assessment generated successfully.")

        except Exception as exc:

            st.error(
                "The inference service could not generate a prediction."
            )

            st.code(
                f"{type(exc).__name__}: {exc}"
            )

    result = st.session_state.get(
        "last_customer_prediction"
    )

    if result:

        render_prediction_result(result)

        with st.expander("Review submitted model features"):

            st.json(
                st.session_state.get(
                    "last_customer_features",
                    {},
                )
            )


# ============================================================
# 8. BATCH ANALYSIS
# ============================================================

def show_batch_analysis():

    render_header(
        "BULK CUSTOMER ANALYSIS",
        "Batch analysis",
        "Upload customer records, validate their schema and generate bulk model predictions.",
    )

    model_ok, model_message = check_model_status()

    if not model_ok:
        st.error(f"Batch prediction unavailable: {model_message}")
        return

    try:
        metadata = load_model_metadata()
    except Exception as exc:
        st.error(f"Unable to load model metadata: {exc}")
        return

    if not metadata:
        st.error("Model metadata is unavailable.")
        return

    feature_columns = metadata.get("feature_columns", [])

    if len(feature_columns) != 17:
        st.error("The model metadata does not contain the expected 17-feature schema.")
        return

    st.info(
        "CSV schema: include all 17 model feature columns. Binary features "
        "must be 0/1; InternetService accepts DSL, Fiber-optic, or No internet "
        "service; Contract and PaymentMethod must use the categories listed "
        "in the data dictionary. Extra columns are excluded from scoring."
    )
    st.caption(
        "Only upload data you are authorized to process. Uploaded records and "
        "scores remain in this app session so you can review and download them. "
        "Avoid customer names, contact details, and other unnecessary identifiers."
    )

    uploaded_file = st.file_uploader(
        "Choose a customer CSV file",
        type=["csv"],
        help="Maximum 10 MB and 10,000 records.",
        key="batch_csv_uploader",
    )

    if uploaded_file is None:
        st.caption(
            "Required features are defined in artifacts/model_metadata.json."
        )
        return

    if uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("File exceeds the 10 MB upload limit.")
        return

    upload_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("last_batch_input_sha256") != upload_digest:
        st.session_state.pop("last_batch_result", None)
        st.session_state["last_batch_input_sha256"] = upload_digest

    try:

        uploaded_file.seek(0)
        batch_df = pd.read_csv(uploaded_file)

    except Exception as exc:

        st.error(
            f"Unable to read CSV: {type(exc).__name__}: {exc}"
        )
        return

    if batch_df.empty:
        st.error("The uploaded CSV contains no records.")
        return

    if len(batch_df) > MAX_BATCH_ROWS:
        st.error(
            f"Maximum batch size is {MAX_BATCH_ROWS:,} records. "
            f"Uploaded file contains {len(batch_df):,}."
        )
        return

    if batch_df.columns.duplicated().any():

        duplicate_columns = (
            batch_df.columns[
                batch_df.columns.duplicated()
            ].tolist()
        )

        st.error(
            f"Duplicate column names: {duplicate_columns}"
        )
        return

    missing_columns = [
        column
        for column in feature_columns
        if column not in batch_df.columns
    ]

    if missing_columns:

        st.error("Required model features are missing:")

        st.code("\n".join(missing_columns))

        return

    extra_columns = [
        column
        for column in batch_df.columns
        if column not in feature_columns
    ]

    if extra_columns:

        st.warning(
            "Extra columns will not be passed to the model: "
            + ", ".join(map(str, extra_columns[:15]))
        )

    model_input = batch_df.loc[
        :,
        feature_columns,
    ].copy()

    st.success(
        f"Loaded {len(model_input):,} customer records. "
        "All 17 required feature columns are present."
    )

    render_section(
        "Input preview",
        "The model will receive only its approved 17 input features.",
    )

    st.dataframe(
        model_input.head(10),
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Validate and score batch",
        type="primary",
        use_container_width=True,
        key="run_batch_prediction",
    ):

        try:

            with st.spinner(
                f"Scoring {len(model_input):,} customer records..."
            ):

                scored_df = predict_batch(model_input)

            st.session_state["last_batch_result"] = scored_df

            st.session_state["last_batch_filename"] = (
                uploaded_file.name
            )

            st.success(
                f"Successfully scored {len(scored_df):,} records."
            )

        except Exception as exc:

            st.session_state.pop(
                "last_batch_result",
                None,
            )

            st.error(
                "Batch validation or inference failed. "
                "No partial results were accepted."
            )

            st.code(
                f"{type(exc).__name__}: {exc}"
            )

    scored_df = st.session_state.get(
        "last_batch_result"
    )

    if isinstance(scored_df, pd.DataFrame):

        render_section(
            "Batch prediction results",
            "Estimated model outcomes for the uploaded customer records.",
        )

        total = len(scored_df)

        predicted_churn = int(
            (scored_df["predicted_class"] == 1).sum()
        )

        predicted_retained = total - predicted_churn

        average_score = float(
            scored_df["churn_probability"].mean()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Scored records",
            f"{total:,}",
        )

        col2.metric(
            "Predicted churn",
            f"{predicted_churn:,}",
        )

        col3.metric(
            "Predicted no churn",
            f"{predicted_retained:,}",
        )

        col4.metric(
            "Average model score",
            f"{average_score:.2%}",
        )

        out_of_range_count = int(
            scored_df["out_of_training_range"].sum()
        )
        if out_of_range_count:
            st.warning(
                f"{out_of_range_count:,} records contain numeric values "
                "outside the model development ranges. Review those scores "
                "before use."
            )

        st.dataframe(
            scored_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = safe_csv_bytes(scored_df)

        st.download_button(
            "Download scored customer CSV",
            data=csv_bytes,
            file_name="churn_batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Predictions are estimates, not verified customer outcomes."
        )


# ============================================================
# 9. INSIGHTS AND REPORTS
# ============================================================

def show_insights():

    render_header(
        "BUSINESS INTELLIGENCE",
        "Insights & reports",
        "Explore observed customer churn patterns and model evaluation experiments.",
    )

    df = load_dataset()

    if df is None:
        st.warning("Source dataset unavailable.")
        return

    required = {
        "Contract",
        "Churn",
        "InternetService",
    }

    if not required.issubset(df.columns):
        st.error("Dataset is missing required insight columns.")
        return

    render_section(
        "Churn by contract",
        "Observed churn rates across contract categories.",
    )

    contract_data = (
        df.groupby("Contract", dropna=False)["Churn"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("Churn rate (%)")
    )

    st.bar_chart(
        contract_data,
        horizontal=True,
    )

    render_section(
        "Churn by internet service",
        "Observed churn rates by internet service category.",
    )

    internet_data = (
        df.assign(
            InternetService=df["InternetService"].fillna(
                "Not recorded"
            )
        )
        .groupby("InternetService", dropna=False)["Churn"]
        .mean()
        .mul(100)
        .rename("Churn rate (%)")
    )

    st.bar_chart(
        internet_data,
        horizontal=True,
    )

    render_section(
        "Threshold experiment",
        "Out-of-fold classification results at different thresholds.",
    )

    threshold_report = load_threshold_report()

    if threshold_report is not None:

        st.dataframe(
            threshold_report,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download threshold report",
            data=threshold_report.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="threshold_analysis.csv",
            mime="text/csv",
        )

    else:

        render_empty_state(
            "Threshold report unavailable",
            "Run the threshold analysis experiment first.",
        )

    st.caption(
        "Group-level churn rates are descriptive associations, "
        "not causal explanations or individual predictions."
    )


# ============================================================
# 10. MODEL HEALTH
# ============================================================

def show_model_health():

    render_header(
        "MODEL GOVERNANCE",
        "Model health",
        "Inspect model identity, evaluation evidence and operational readiness.",
    )

    try:
        metadata = load_model_metadata()
    except Exception as exc:
        metadata = None
        st.error(f"Metadata error: {exc}")

    model_ok, model_message = check_model_status()

    if model_ok:
        st.success(
            "Local model artifact is available and can be loaded."
        )
    else:
        st.error(model_message)

    if metadata:

        render_section("Model identity")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Model",
            str(metadata.get("model_name", "Unknown")),
        )

        col2.metric(
            "Version",
            str(metadata.get("model_version", "Unknown")),
        )

        training_rows = metadata.get("training_rows")

        col3.metric(
            "Training rows",
            f"{training_rows:,}"
            if isinstance(training_rows, int)
            else "Unknown",
        )

        details = {
            "Artifact": metadata.get("artifact_filename"),
            "Created at UTC": metadata.get("created_at_utc"),
            "Python": metadata.get("python_version"),
            "Scikit-learn": metadata.get("scikit_learn_version"),
            "Target": metadata.get("target_column"),
            "Training data SHA-256": metadata.get("training_data_sha256"),
        }

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Property": key,
                        "Value": value,
                    }
                    for key, value in details.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        render_section(
            "Decision threshold",
            "The current threshold is stored in model metadata.",
        )

        threshold = metadata.get("threshold")

        st.metric(
            "Technical threshold",
            f"{float(threshold):.2f}"
            if threshold is not None
            else "Unavailable",
        )

        st.warning(
            metadata.get(
                "threshold_status",
                "Threshold approval status unavailable.",
            )
        )

        render_section(
            "Held-out test performance",
            "Metrics recorded during the final model export.",
        )

        metrics = metadata.get(
            "final_test_metrics",
            {},
        )

        if metrics:

            metric_cols = st.columns(4)

            for column, (label, key) in zip(
                metric_cols,
                [
                    ("Accuracy", "accuracy"),
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1-score", "f1"),
                ],
            ):

                column.metric(
                    label,
                    format_percentage(metrics.get(key)),
                )

            metric_cols_2 = st.columns(4)

            for column, (label, key) in zip(
                metric_cols_2,
                [
                    ("ROC-AUC", "roc_auc"),
                    ("PR-AUC", "pr_auc"),
                    ("Brier score", "brier_score"),
                    ("Test records", "test_rows"),
                ],
            ):

                value = metrics.get(key)

                if key == "test_rows":
                    display_value = (
                        f"{value:,}"
                        if isinstance(value, int)
                        else "Unavailable"
                    )
                elif key == "brier_score":
                    display_value = (
                        f"{float(value):.3f}"
                        if value is not None
                        else "Unavailable"
                    )
                else:
                    display_value = format_percentage(value)

                column.metric(label, display_value)

            render_section(
                "Held-out test confusion matrix",
            )

            confusion = pd.DataFrame(
                {
                    "Outcome": [
                        "Actual retained",
                        "Actual churned",
                    ],
                    "Predicted retained": [
                        metrics.get("true_negatives"),
                        metrics.get("false_negatives"),
                    ],
                    "Predicted churned": [
                        metrics.get("false_positives"),
                        metrics.get("true_positives"),
                    ],
                }
            ).set_index("Outcome")

            st.dataframe(
                confusion,
                use_container_width=True,
            )

        limitations = metadata.get("limitations", [])

        if limitations:

            render_section("Known limitations")

            for limitation in limitations:
                st.markdown(f"- {limitation}")

    render_section(
        "Evaluation reports",
        "Development cross-validation and threshold experiment outputs.",
    )

    cv_report = load_cv_report()
    threshold_report = load_threshold_report()

    if cv_report is not None:

        with st.expander(
            "Cross-validation report",
            expanded=False,
        ):

            st.dataframe(
                cv_report,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download cross-validation report",
                data=cv_report.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="cv_model_comparison.csv",
                mime="text/csv",
                key="download_cv",
            )

    else:
        st.warning("Cross-validation report unavailable.")

    if threshold_report is not None:

        with st.expander(
            "Threshold analysis report",
            expanded=False,
        ):

            st.dataframe(
                threshold_report,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download threshold report",
                data=threshold_report.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="threshold_analysis.csv",
                mime="text/csv",
                key="download_threshold",
            )

    else:
        st.warning("Threshold report unavailable.")

    subgroup_report = load_subgroup_report()
    if subgroup_report is not None:
        with st.expander(
            "Descriptive subgroup performance review",
            expanded=False,
        ):
            st.warning(
                "These aggregate metrics are diagnostic only. They do not "
                "establish fairness, statistical significance, or legal compliance."
            )
            st.dataframe(
                subgroup_report,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download subgroup review",
                data=subgroup_report.to_csv(index=False).encode("utf-8"),
                file_name="subgroup_performance.csv",
                mime="text/csv",
                key="download_subgroup_report",
            )
    else:
        st.info(
            "No subgroup review report is available. Run "
            "scripts/subgroup_audit.py to generate diagnostic metrics."
        )

    render_section("Operational readiness")

    readiness = pd.DataFrame(
        {
            "Component": [
                "Source dataset",
                "Cross-validation reports",
                "Threshold experiments",
                "Subgroup review report",
                "Model artifact",
                "Model metadata",
                "Local inference service",
                "Production deployment",
                "Prediction monitoring",
                "Data drift monitoring",
                "Automated rollback",
            ],
            "Status": [
                "Available" if DATA_PATH.is_file() else "Missing",
                "Available" if CV_REPORT_PATH.is_file() else "Missing",
                "Available" if THRESHOLD_REPORT_PATH.is_file() else "Missing",
                "Available" if SUBGROUP_REPORT_PATH.is_file() else "Missing",
                "Available" if MODEL_PATH.is_file() else "Missing",
                "Available" if METADATA_PATH.is_file() else "Missing",
                "Available" if model_ok else "Unavailable",
                "Not implemented",
                "Not implemented",
                "Not implemented",
                "Not implemented",
            ],
        }
    )

    st.dataframe(
        readiness,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The model is exported and usable for local inference. "
        "Production deployment, monitoring, drift detection, "
        "threshold approval, alerting and rollback remain separate tasks."
    )


# ============================================================
# 11. PAGE ROUTING
# ============================================================

PAGE_RENDERERS = {
    "Overview": show_overview,
    "Predict Customer": show_prediction,
    "Batch Analysis": show_batch_analysis,
    "Insights & Reports": show_insights,
    "Model Health": show_model_health,
}

PAGE_RENDERERS[selected_page]()

render_footer()
