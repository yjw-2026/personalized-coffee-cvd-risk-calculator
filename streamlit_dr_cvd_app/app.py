from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "dr_model.pkl"
CONFIG_PATH = APP_DIR / "deploy_config.json"
EXAMPLE_INPUT_PATH = APP_DIR / "examples" / "example_input.csv"
EXAMPLE_OUTPUT_PATH = APP_DIR / "examples" / "example_output_with_rd.csv"

st.set_page_config(
    page_title="Personalized CVD Risk Difference Calculator",
    page_icon="☕",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_examples():
    example_input = pd.read_csv(EXAMPLE_INPUT_PATH)
    example_output = pd.read_csv(EXAMPLE_OUTPUT_PATH)
    return example_input, example_output


def predict_cate_rd(est, x_raw: pd.DataFrame) -> np.ndarray:
    """Predict individual RD using the saved DR-Learner and its attached preprocessor."""
    if not hasattr(est, "_x_pre"):
        raise AttributeError(
            "The loaded model does not contain est._x_pre. "
            "Please export the model after attaching the fitted preprocessor to est._x_pre."
        )
    x_num = est._x_pre.transform(x_raw)
    return np.asarray(est.effect(x_num)).ravel()


def build_input_dataframe(config: dict) -> pd.DataFrame:
    cont_vars = config["cont_vars"]
    cat_vars = config["cat_vars"]
    categories = config["categories"]

    st.subheader("Enter covariates")
    st.caption("The input variables and category levels are fixed to match the original model training code.")

    values = {}

    with st.expander("Continuous variables", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            values["Age.at.recruitment"] = st.number_input("Age at recruitment", min_value=18.0, max_value=100.0, value=60.0, step=1.0)
            values["Sleep.duration"] = st.number_input("Sleep duration (hours/day)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            values["Tea.intake"] = st.number_input("Tea intake", min_value=0.0, max_value=30.0, value=1.0, step=0.5)
        with c2:
            values["TDI"] = st.number_input("Townsend deprivation index (TDI)", min_value=-10.0, max_value=20.0, value=-2.0, step=0.1)
            values["BMI"] = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)

    with st.expander("Categorical variables", expanded=True):
        for col in cat_vars:
            default_idx = 0
            values[col] = st.selectbox(col, categories[col], index=default_idx)

    # Keep exact training-time column order.
    row = {col: values[col] for col in cont_vars + cat_vars}
    return pd.DataFrame([row])


def show_result(rd: float):
    rd_pct = rd * 100
    st.subheader("Predicted result")
    st.metric("Individualized risk difference (RD)", f"{rd_pct:.2f} percentage points")

    if rd < 0:
        st.success("Interpretation: the model predicts a lower CVD risk under coffee intake for this covariate profile.")
    elif rd > 0:
        st.warning("Interpretation: the model predicts a higher CVD risk under coffee intake for this covariate profile.")
    else:
        st.info("Interpretation: the model predicts no difference in CVD risk for this covariate profile.")

    st.caption(
        "RD is calculated as the model-estimated CVD risk difference for coffee intake versus no coffee intake. "
        "Negative values indicate lower predicted CVD risk under coffee intake."
    )


def main():
    st.title("Personalized CVD Risk Difference Calculator")
    st.write(
        "This prototype uses a saved DR-Learner model to estimate an individualized CVD risk difference "
        "based on user-entered covariates."
    )

    with st.info("Research-use prototype. This tool is not intended for clinical diagnosis, treatment decisions, or individual medical advice."):
        pass

    config = load_config()

    with st.sidebar:
        st.header("Model information")
        st.write("Model: DR-Learner")
        st.write("Training: 70% training set")
        st.write("Output: individualized RD")
        st.divider()
        st.write("Use the example button to populate the form indirectly via a preview below; the current manual inputs are used for prediction.")

    x_new = build_input_dataframe(config)

    st.subheader("Input preview")
    st.dataframe(x_new, use_container_width=True)

    if st.button("Predict RD", type="primary"):
        try:
            est = load_model()
            rd = float(predict_cate_rd(est, x_new)[0])
            show_result(rd)
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
            st.write(
                "Common causes include missing Python packages such as econml, version mismatch with scikit-learn/econml, "
                "or an incorrectly exported model object."
            )

    st.divider()
    st.subheader("Example data for verification")
    try:
        example_input, example_output = load_examples()
        st.write("Example input rows exported from the training data:")
        st.dataframe(example_input, use_container_width=True)
        st.write("Expected RD outputs for these rows:")
        st.dataframe(example_output[[*config["cont_vars"], *config["cat_vars"], "RD_pred"]], use_container_width=True)
    except Exception as e:
        st.warning(f"Example files could not be loaded: {e}")

    st.divider()
    st.subheader("Important notes")
    st.write(
        "The model estimates individualized associations/effects under the assumptions of the original DR-Learner analysis. "
        "It should be interpreted as a research demonstration rather than a validated clinical risk calculator."
    )


if __name__ == "__main__":
    main()
