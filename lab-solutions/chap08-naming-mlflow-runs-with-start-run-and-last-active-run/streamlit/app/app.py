import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("Chapter 08 — start_run(run_name=...) + last_active_run()")
st.caption("Today's topic: giving runs a human-readable name and querying the last one.")

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
        st.success(f"Run **{data['run_name']}** finished with status **{data['status']}**")
        st.json(data)
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "Open the [MLflow UI](http://localhost:5000), select **`wine_quality_chap08`**, "
    "and notice that runs now have the names you typed instead of auto-generated ones."
)
