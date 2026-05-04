import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")
st.set_page_config(page_title="Chap 17 - Registry", page_icon=":file_folder:")
st.title("Chapter 17 - Model Registry")

tab_train, tab_promote, tab_predict = st.tabs(
    ["1) Train + Register", "2) Promote a version", "3) Predict (Production)"]
)

with tab_train:
    a = st.number_input("alpha", 0.0, 1.0, 0.4, 0.1, key="reg_a")
    l = st.number_input("l1_ratio", 0.0, 1.0, 0.4, 0.1, key="reg_l")
    if st.button("Train + Register", type="primary"):
        try:
            r = requests.post(
                f"{API_URL}/train-and-register",
                json={"alpha": a, "l1_ratio": l},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
            st.success(
                f"Registered as {data['registered']['name']} "
                f"v{data['registered']['version']} "
                f"(stage: {data['registered']['current_stage']})"
            )
            st.json(data)
        except requests.RequestException as e:
            st.error(f"Failed: {e}")

with tab_promote:
    if st.button("Refresh versions list"):
        try:
            st.session_state["versions"] = requests.get(
                f"{API_URL}/versions", timeout=30
            ).json()
        except requests.RequestException as e:
            st.error(f"Failed: {e}")
    if "versions" in st.session_state:
        st.json(st.session_state["versions"])
    v = st.number_input("Version to promote", 1, 999, 1, 1, key="prom_v")
    s = st.selectbox(
        "New stage", ["Staging", "Production", "Archived", "None"], key="prom_s"
    )
    archive = st.checkbox(
        "Archive other versions in same stage", value=True, key="prom_arch"
    )
    if st.button("Promote"):
        try:
            r = requests.post(
                f"{API_URL}/promote",
                json={"version": v, "stage": s, "archive_existing": archive},
                timeout=60,
            )
            r.raise_for_status()
            st.success(f"Version {v} -> {s}")
            st.json(r.json())
        except requests.RequestException as e:
            st.error(f"Failed: {e}")

with tab_predict:
    sample = {
        "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4,
    }
    payload_text = st.text_area(
        "JSON payload (a list of rows)",
        value=json.dumps([sample], indent=2),
        height=280,
    )
    if st.button("Predict via models:/.../Production"):
        try:
            r = requests.post(
                f"{API_URL}/predict-production",
                json={"rows": json.loads(payload_text)},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            st.success(f"Resolved URI: {data['model_uri']}")
            for i, p in enumerate(data["predictions"]):
                st.metric(f"Row {i + 1}", f"{p}")
            with st.expander("Full response"):
                st.json(data)
        except requests.HTTPError as e:
            st.error(f"{e} - did you promote a version to Production?")
        except (requests.RequestException, json.JSONDecodeError) as e:
            st.error(f"Failed: {e}")
