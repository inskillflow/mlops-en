import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("MLOps Demo — Streamlit + FastAPI + MLflow")

st.write(f"API URL: `{API_URL}`")

features_text = st.text_input("Features (comma separated)", "1.0, 2.0, 3.0")

if st.button("Send to FastAPI"):
    try:
        features = [float(x.strip()) for x in features_text.split(",")]
        r = requests.post(f"{API_URL}/log-run", json={"features": features}, timeout=10)
        st.success("Logged!")
        st.json(r.json())
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown("**MLflow UI** — open [http://localhost:5000](http://localhost:5000)")
