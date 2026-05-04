import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.set_page_config(page_title="Chap 12 - sweeps", page_icon=":repeat:")
st.title("Chapter 12 - multiple runs & multiple experiments")

st.markdown(
    "Two batch endpoints:\n"
    "- **Grid** runs 9 ElasticNet configs in one experiment.\n"
    "- **Sweep** runs 3 alphas x 3 model families = **3 experiments x 3 runs**."
)

c1, c2 = st.columns(2)

if c1.button("Run grid (9 runs)", type="primary"):
    with st.spinner("Training 9 ElasticNet runs..."):
        try:
            r = requests.post(f"{API_URL}/train-grid", timeout=600)
            r.raise_for_status()
            data = r.json()
            st.success(f"{data['n_runs']} runs logged in `{data['experiment']}`")
            st.markdown(
                f"**Best run** (lowest RMSE): `{data['best']['run_name']}` "
                f"alpha={data['best']['alpha']}, "
                f"l1_ratio={data['best']['l1_ratio']}, "
                f"RMSE={data['best']['rmse']:.4f}"
            )
            st.dataframe(data["all"])
        except requests.RequestException as e:
            st.error(f"Failed: {e}")

if c2.button("Run sweep (9 runs / 3 experiments)"):
    with st.spinner("Training 3 model families..."):
        try:
            r = requests.post(f"{API_URL}/train-sweep", timeout=600)
            r.raise_for_status()
            data = r.json()
            st.success(
                f"{data['n_runs']} runs across {data['n_experiments']} experiments"
            )
            best = data["overall_best"]
            st.markdown(
                f"**Overall best**: family **{best['family']}** with "
                f"alpha={best['alpha']}, RMSE={best['rmse']:.4f}"
            )
            for family, summary in data["by_family"].items():
                with st.expander(
                    f"{family} - best alpha={summary['best']['alpha']} "
                    f"(RMSE={summary['best']['rmse']:.4f})"
                ):
                    st.dataframe(summary["all"])
        except requests.RequestException as e:
            st.error(f"Failed: {e}")
