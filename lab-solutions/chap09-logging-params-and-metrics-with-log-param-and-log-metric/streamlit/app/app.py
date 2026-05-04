import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("Chapter 09 — log_param() + log_metric()")
st.caption("Today's topic: recording hyperparameters and evaluation metrics in each run.")

st.write(f"API URL: `{API_URL}`")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", value=0.5, step=0.1)
l1_ratio = col2.number_input("l1_ratio", value=0.5, step=0.1)

run_name = st.text_input("Run name (optional)", placeholder="e.g. baseline_v1")

if st.button("Train"):
    payload = {"alpha": alpha, "l1_ratio": l1_ratio}
    if run_name.strip():
        payload["run_name"] = run_name.strip()
    try:
        r = requests.post(f"{API_URL}/train", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        st.success(f"Run **{data['run_name']}** done")

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{data['metrics']['rmse']:.4f}")
        c2.metric("MAE", f"{data['metrics']['mae']:.4f}")
        c3.metric("R²", f"{data['metrics']['r2']:.4f}")

        with st.expander("Full response"):
            st.json(data)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "Open the [MLflow UI](http://localhost:5000) → experiment **`wine_quality_chap09`**. "
    "Each run now shows **2 params** (alpha, l1_ratio) and **3 metrics** (rmse, mae, r2)."
)
