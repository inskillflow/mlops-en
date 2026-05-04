import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.set_page_config(page_title="Chap 14 - signature", page_icon=":memo:")
st.title("Chapter 14 - signature + input_example")
st.caption(
    "Train an ElasticNet model. The signature and an input_example "
    "(5 rows) are saved with the model."
)

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

        with st.expander("Model signature attached to the run"):
            st.code(data["signature"], language="text")

        with st.expander("Input example (5 rows)"):
            st.dataframe(data["input_example"])
            st.markdown(
                "Copy any row -> POST it to the model server: "
                "`mlflow models serve -m runs:/<run_id>/model -p 1234 --no-conda`"
            )
    except requests.RequestException as e:
        st.error(f"Failed: {e}")
