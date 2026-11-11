from __future__ import annotations

import streamlit as st
import pandas as pd

from src.data.loader import load_csv
from src.data.validator import validate_classification_data
from src.data.explorer import dataset_profile
from src.training.workflow import train_and_compare
from src.prediction.service import predict_rows
from src.persistence.artifacts import save_experiment, load_model_bundle

st.set_page_config(page_title="ML Workbench", page_icon="🧪", layout="wide")
st.title("🧪 Machine Learning Workbench")
st.caption("A modular, classification-only workflow: inspect → prepare → train → evaluate → save → predict.")

with st.sidebar:
    st.header("Project controls")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    st.caption("Start with Titanic or another tabular classification dataset.")

if uploaded is None:
    st.info("Upload a CSV file to begin. The app keeps the ML logic in the `src/` package; Streamlit is only the interface.")
    st.subheader("What this workbench does")
    st.markdown("""
    1. Checks the dataset and displays a data profile.
    2. Lets you choose the target and feature columns.
    3. Splits data before learning preprocessing rules.
    4. Trains Logistic Regression, Decision Tree, and Random Forest.
    5. Compares models using cross-validation on training data, then evaluates the selected model on the held-out test set.
    6. Saves the fitted preprocessing + model pipeline for future predictions.
    """)
    st.stop()

try:
    df = load_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

st.subheader("1 · Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
st.dataframe(df.head(20), use_container_width=True)

profile = dataset_profile(df)
with st.expander("Column profile", expanded=False):
    st.dataframe(profile, use_container_width=True)
with st.expander("Summary statistics", expanded=False):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

st.subheader("2 · Configure the classification task")
target = st.selectbox("Target column (the value to predict)", options=list(df.columns))
available_features = [c for c in df.columns if c != target]
features = st.multiselect("Input features", options=available_features, default=available_features)
test_size = st.slider("Test set proportion", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
cv_folds = st.select_slider("Cross-validation folds", options=[3, 4, 5], value=5)
tune = st.checkbox("Run a small hyperparameter search (slower)", value=False)

if st.button("Train and compare classifiers", type="primary", disabled=not features):
    try:
        validate_classification_data(df, target, features)
        with st.spinner("Splitting data, fitting pipelines, and evaluating models..."):
            result = train_and_compare(
                df=df, target=target, features=features,
                test_size=test_size, cv_folds=cv_folds, tune=tune
            )
        st.session_state["workbench_result"] = result
        st.session_state["workbench_df"] = df
        st.session_state["workbench_target"] = target
        st.session_state["workbench_features"] = features
        st.success("Training and evaluation completed.")
    except Exception as exc:
        st.error(f"Training could not complete: {exc}")

result = st.session_state.get("workbench_result")
if result is not None:
    st.subheader("3 · Model comparison")
    st.write("Cross-validation scores are calculated on the training data. The test set is held back until final evaluation.")
    st.dataframe(result["comparison"], use_container_width=True)
    st.markdown(f"**Selected model:** `{result['selected_model']}`")
    st.caption(f"Selection rule: highest mean cross-validation F1 score ({result['selection_metric']}).")
    st.subheader("4 · Held-out test evaluation")
    st.dataframe(result["test_metrics"], use_container_width=True)
    st.markdown("**Confusion matrix — selected model**")
    st.dataframe(result["confusion_matrix"], use_container_width=True)
    with st.expander("Classification report"):
        st.dataframe(result["classification_report"], use_container_width=True)
    with st.expander("Best parameters"):
        st.json(result["best_params"])

    if st.button("Save selected model and experiment"):
        model_path, report_path = save_experiment(result)
        st.success(f"Saved model: {model_path} · Report: {report_path}")

    st.subheader("5 · Predict on new rows")
    st.write("Upload a CSV containing the same input feature columns. Do not include the target column.")
    prediction_file = st.file_uploader("Upload prediction rows", type=["csv"], key="prediction_csv")
    if prediction_file is not None:
        try:
            new_rows = load_csv(prediction_file)
            predictions = predict_rows(result["pipeline"], new_rows, result["features"])
            st.dataframe(predictions, use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                data=predictions.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("6 · Predict using a previously saved model")
st.caption("This section loads the latest saved pipeline from the local artifacts folder.")
bundle = load_model_bundle()
if bundle is not None:
    st.write(f"Saved model: **{bundle['metadata'].get('selected_model', 'Unknown')}**")
    saved_file = st.file_uploader("Upload rows for the saved model", type=["csv"], key="saved_prediction_csv")
    if saved_file is not None:
        try:
            rows = load_csv(saved_file)
            output = predict_rows(bundle["pipeline"], rows, bundle["metadata"]["features"])
            st.dataframe(output, use_container_width=True)
            st.download_button("Download saved-model predictions", output.to_csv(index=False).encode(), "saved_model_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not predict with saved model: {exc}")
else:
    st.info("No saved model found yet. Train a model and click 'Save selected model and experiment'.")
