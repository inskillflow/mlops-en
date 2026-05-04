import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.set_page_config(page_title="Chap 11 - tags", page_icon=":label:")
st.title("Chapter 11 - set_tag / set_tags")
st.caption("Each run carries 6 tags. The triggered_by tag is dynamic and chosen below.")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", 0.0, 1.0, 0.5, 0.1)
l1_ratio = col2.number_input("l1_ratio", 0.0, 1.0, 0.5, 0.1)
run_name = st.text_input("Run name (optional)")
triggered_by = st.selectbox(
    "triggered_by",
    options=["streamlit-ui", "curl-cli", "scheduled-job", "data-scientist-A", "data-scientist-B"],
)

if st.button("Train", type="primary"):
    payload = {"alpha": alpha, "l1_ratio": l1_ratio, "triggered_by": triggered_by}
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

        with st.expander("Tags attached to this run"):
            st.json(data["tags"])
            st.markdown(
                "Filter these in the MLflow UI search bar with: "
                f"`tags.triggered_by = \"{data['tags']['triggered_by']}\"`"
            )

        with st.expander("Full response"):
            st.json(data)
    except requests.RequestException as e:
        st.error(f"Failed to call FastAPI: {e}")
