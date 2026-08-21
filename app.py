"""
Advertising Sales Predictor — Streamlit app.

Loads the pre-trained Random Forest model (best_random_forest_advertising.joblib)
and lets the user predict Sales from TV / Radio / Newspaper budget sliders.

Run locally:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advertising Sales Predictor",
    page_icon="📈",
    layout="centered",
)

MODEL_PATH = "best_random_forest_advertising.joblib"
FEATURES = ["TV", "Radio", "Newspaper"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Make sure the joblib file is in the "
        "same directory as app.py (both locally and in your GitHub repo)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Advertising Sales Predictor")
st.write(
    "Adjust the ad spend budgets below to predict Sales using a trained "
    "Random Forest model (R² ≈ 0.98 on held-out test data)."
)

st.divider()

# ---------------------------------------------------------------------------
# Inputs — sliders in a sidebar so the result stays visible
# ---------------------------------------------------------------------------
st.sidebar.header("Ad spend budgets ($ thousands)")

tv = st.sidebar.slider("TV budget", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
radio = st.sidebar.slider("Radio budget", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
newspaper = st.sidebar.slider("Newspaper budget", min_value=0.0, max_value=115.0, value=30.0, step=1.0)

input_df = pd.DataFrame([[tv, radio, newspaper]], columns=FEATURES)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
prediction = model.predict(input_df)[0]

col1, col2 = st.columns([2, 1])
with col1:
    st.metric(label="Predicted Sales (thousands of units)", value=f"{prediction:.2f}")
with col2:
    st.write("")  # spacing
    st.write(f"**Total budget:** ${tv + radio + newspaper:,.1f}k")

st.divider()

# ---------------------------------------------------------------------------
# Feature importance context
# ---------------------------------------------------------------------------
st.subheader("What's driving this prediction?")

importances = pd.DataFrame(
    {"Channel": FEATURES, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)

st.bar_chart(importances.set_index("Channel"))

st.caption(
    "Feature importance reflects the trained model overall, not this specific "
    "prediction — TV and Radio drive most of the model's predictive power in "
    "this dataset; Newspaper spend has minimal impact."
)

# ---------------------------------------------------------------------------
# Raw input summary (helpful for debugging / screenshots)
# ---------------------------------------------------------------------------
with st.expander("See input values sent to the model"):
    st.dataframe(input_df, hide_index=True)