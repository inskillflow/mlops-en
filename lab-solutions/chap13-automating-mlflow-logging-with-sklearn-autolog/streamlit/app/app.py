import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.set_page_config(page_title="Chap 13 - autolog", page_icon=":robot_face:")
st.title("Chapter 13 - mlflow.sklearn.autolog()")
st.caption("Train an ElasticNet model. Hyperparameters, training metrics and the model itself are auto-logged.")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", 0.0, 1.0, 0.5, 0.1)
l1_ratio = col2.number_input("l1_ratio", 0.0, 1.0, 0.5, 0.1)
run_name = st.text_input("Run name (optional)")

if st.button("Train", type="primary"):
    payload = {"alpha": alpha, "l1_ratio": l1_ratio}
    if run_name.strip():
        payload["run_name"] = run_name.strip()
    try:
        r = requests.post(f"{API_URL}/train", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        st.success(f"Run `{data['run_name']}` created in `{data['experiment']}`")

        m = data["test_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Test RMSE", f"{m['rmse']:.4f}")
        c2.metric("Test MAE", f"{m['mae']:.4f}")
        c3.metric("Test R2", f"{m['r2']:.4f}")

        st.info(data["note"])
        with st.expander("Full response"):
            st.json(data)
    except requests.RequestException as e:
        st.error(f"Failed: {e}")
