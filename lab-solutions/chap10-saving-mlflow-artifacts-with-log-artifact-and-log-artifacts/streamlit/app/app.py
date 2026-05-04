import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")
MLFLOW_PUBLIC_URL = os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5000")

st.set_page_config(page_title="Chap 10 - artifacts", page_icon=":bar_chart:")
st.title("Chapter 10 - log_artifact / log_artifacts")
st.caption("Train an ElasticNet model. The CSV of predictions and a scatter plot are logged as artifacts.")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
l1_ratio = col2.number_input("l1_ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
run_name = st.text_input("Optional run name", value="")

if st.button("Train", type="primary"):
    payload = {"alpha": alpha, "l1_ratio": l1_ratio}
    if run_name.strip():
        payload["run_name"] = run_name.strip()
    try:
        r = requests.post(f"{API_URL}/train", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()

        st.success(f"Run `{data['run_name']}` created in `{data['experiment']}`")

        m = data["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{m['rmse']:.4f}")
        c2.metric("MAE", f"{m['mae']:.4f}")
        c3.metric("R2", f"{m['r2']:.4f}")

        with st.expander("Artifact URI"):
            st.code(data["artifact_uri"])
            st.markdown(
                f"Open the run in MLflow UI: [{MLFLOW_PUBLIC_URL}]({MLFLOW_PUBLIC_URL})"
            )

        with st.expander("Full response"):
            st.json(data)
    except requests.RequestException as e:
        st.error(f"Failed to call FastAPI: {e}")
