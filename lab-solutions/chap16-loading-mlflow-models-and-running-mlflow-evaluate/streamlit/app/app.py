import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.set_page_config(page_title="Chap 16 - load + evaluate", page_icon=":mag:")
st.title("Chapter 16 - load_model + mlflow.evaluate")

tab_train, tab_predict = st.tabs(["1) Train + evaluate", "2) Predict from run"])

with tab_train:
    col1, col2 = st.columns(2)
    alpha = col1.number_input("alpha", 0.0, 1.0, 0.4, 0.1, key="t_a")
    l1_ratio = col2.number_input("l1_ratio", 0.0, 1.0, 0.4, 0.1, key="t_l")
    run_name = st.text_input("Run name (optional)", key="t_n")
    if st.button("Train + evaluate", type="primary"):
        payload = {"alpha": alpha, "l1_ratio": l1_ratio}
        if run_name.strip():
            payload["run_name"] = run_name.strip()
        try:
            r = requests.post(
                f"{API_URL}/train-and-evaluate", json=payload, timeout=300
            )
            r.raise_for_status()
            data = r.json()
            st.success(f"Run `{data['run_name']}`")
            st.markdown("**run_id (copy for the Predict tab):**")
            st.code(data["run_id"])
            st.markdown("**Manual metrics**")
            st.json(data["manual_metrics"])
            st.markdown("**`mlflow.evaluate` metrics**")
            st.json(data["evaluate_metrics"])
        except requests.RequestException as e:
            st.error(f"Failed: {e}")

with tab_predict:
    run_id = st.text_input("run_id (paste from the train tab)")
    sample = {
        "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4,
    }
    payload_text = st.text_area(
        "JSON payload (a list of rows)",
        value=json.dumps([sample], indent=2),
        height=300,
    )
    if st.button("Predict"):
        try:
            rows = json.loads(payload_text)
            r = requests.post(
                f"{API_URL}/predict",
                json={"run_id": run_id.strip(), "rows": rows},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            st.success(f"{data['n_rows']} predictions")
            for i, p in enumerate(data["predictions"]):
                st.metric(f"Row {i + 1}", f"{p}")
            with st.expander("Full response"):
                st.json(data)
        except requests.RequestException as e:
            st.error(f"Failed: {e}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
