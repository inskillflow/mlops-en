import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("Chapter 07 — mlflow.set_experiment()")
st.caption("Today's topic: creating or selecting an MLflow experiment.")

st.write(f"API URL: `{API_URL}`")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", value=0.5, step=0.1)
l1_ratio = col2.number_input("l1_ratio", value=0.5, step=0.1)

if st.button("Train"):
    try:
        r = requests.post(
            f"{API_URL}/train",
            json={"alpha": alpha, "l1_ratio": l1_ratio},
            timeout=60,
        )
        r.raise_for_status()
        st.success("Run created in the experiment")
        st.json(r.json())
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "Open the [MLflow UI](http://localhost:5000) and look for the experiment "
    "**`wine_quality_chap07`**. Each click on **Train** creates a new (empty) run inside it."
)
