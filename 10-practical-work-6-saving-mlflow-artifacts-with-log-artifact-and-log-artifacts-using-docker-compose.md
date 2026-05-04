<a id="top"></a>

# Chapter 10 — Today's topic: `mlflow.log_artifact()` + `mlflow.log_artifacts()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What is an artifact?](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The lines we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Updated `streamlit/app/app.py` (show artifact URI)](#section-7) |
| 8 | [Run the stack](#section-8) |
| 9 | [Visualize artifacts in the MLflow UI](#section-9) |
| 10 | [Mini exercise — log a third artifact](#section-10) |
| 11 | [`log_artifact` vs `log_artifacts` — when to use which](#section-11) |
| 12 | [Tear down](#section-11b) |
| 13 | [Recap](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack as Chapters 07 → 09 and we add **three new MLflow lines**:

- `mlflow.log_artifact(path)` — attach **one file** (a CSV, an image, a JSON…) to the current run;
- `mlflow.log_artifacts(folder, artifact_path=...)` — attach **a whole folder** in one shot;
- and we generate a small **scatter plot** (`predictions_plot.png`) and a **CSV of predictions** (`test_predictions.csv`) so we have something concrete to log.

After this chapter, every run in the MLflow UI carries its own **downloadable evidence**: the plot, the predictions, the data folder. You can re-open a run from last week and click **Download** on any artifact.

> [!IMPORTANT]
> Up to Chapter 09, runs were just numbers. From now on, runs become **full experiment records**: numbers + figures + raw outputs, all reproducible.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What is an artifact?

An **artifact** is *any file produced by your script that you want to keep with the run*.

| Examples of artifacts | Type |
|---|---|
| `test_predictions.csv` | Tabular output |
| `predictions_plot.png` | Visualization |
| `model.pkl`, `model.joblib` | Serialized model (we'll see `log_model` later) |
| `report.html`, `report.pdf` | Auto-generated reports |
| `train.log` | Training logs |
| A whole folder of feature CSVs | Bulk export |

Artifacts are stored in the **artifact store** of MLflow. In our stack, that's the named volume `mlflow-artifacts` mounted at `/mlflow/mlruns` inside the `mlflow` container — exactly the place we set up in Chapter 06.

> [!NOTE]
> Artifacts are **immutable** once logged: you cannot edit `predictions_plot.png` of a finished run. To change it, you start a **new run**. That is the whole point — perfect reproducibility.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.log_artifact(local_path, artifact_path=None)` | **NEW today.** Logs **one file** to the current run. |
| `mlflow.log_artifacts(local_dir, artifact_path=None)` | **NEW today.** Logs **all files** of `local_dir` (recursively). |
| `mlflow.get_artifact_uri(artifact_path=None)` | (Bonus) Returns the URI where artifacts are stored, e.g. `mlflow-artifacts:/<exp>/<run>/artifacts`. Useful in API responses. |

> [!IMPORTANT]
> Notice the **plural / singular** difference: `log_artifact` (file) vs `log_artifacts` (folder). It's a one-letter difference but a totally different signature. Forgetting the `s` and passing a folder will silently log only the folder's name as an empty file.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The lines we add today

Three new lines, plus a few helper lines that **build the artifacts** in a temporary folder first:

```python
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with tempfile.TemporaryDirectory() as tmpdir:
    csv_path = os.path.join(tmpdir, "test_predictions.csv")
    pd.DataFrame({"actual": test_y.values, "predicted": preds}).to_csv(csv_path, index=False)

    fig, ax = plt.subplots()
    ax.scatter(test_y, preds, alpha=0.5)
    ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], "r--", linewidth=1)
    ax.set_xlabel("actual"); ax.set_ylabel("predicted")
    png_path = os.path.join(tmpdir, "predictions_plot.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(csv_path)                           # NEW
    mlflow.log_artifact(png_path)                           # NEW
    mlflow.log_artifacts(tmpdir, artifact_path="all_outputs")  # NEW
```

Why a `tempfile.TemporaryDirectory()`? Because the FastAPI container has a read-only `/code` folder and we don't want to leave clutter on disk between runs. `tempfile` creates a folder, MLflow uploads its content to the artifact store, then the temp folder is deleted automatically when the `with` block exits.

> [!NOTE]
> `matplotlib.use("Agg")` is **required** in a server (no display). Without it, matplotlib tries to open a GUI window and the FastAPI process crashes inside the container.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap10-saving-mlflow-artifacts-with-log-artifact-and-log-artifacts/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt          ← matplotlib added
│   └── app/
│       └── main.py               ← changes today
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py                ← shows the artifact URI
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Modified code — `fastapi/app/main.py`

```python
import io
import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI)
import matplotlib.pyplot as plt

import mlflow
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap10"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 10 — log_artifact / log_artifacts")


def load_data() -> pd.DataFrame:
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep=";")


def eval_metrics(actual, pred):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2


class TrainRequest(BaseModel):
    alpha: float = 0.5
    l1_ratio: float = 0.5
    run_name: str | None = None


@app.post("/train")
def train(req: TrainRequest):
    data = load_data()
    train_df, test_df = train_test_split(data, random_state=40)
    train_x = train_df.drop(["quality"], axis=1)
    test_x = test_df.drop(["quality"], axis=1)
    train_y = train_df["quality"]
    test_y = test_df["quality"]

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = req.run_name or f"alpha_{req.alpha}_l1_{req.l1_ratio}"

    with mlflow.start_run(run_name=run_name):
        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        preds = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)

        mlflow.log_param("alpha", req.alpha)
        mlflow.log_param("l1_ratio", req.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        with tempfile.TemporaryDirectory() as tmpdir:                     # NEW
            csv_path = os.path.join(tmpdir, "test_predictions.csv")        # NEW
            pd.DataFrame(                                                  # NEW
                {"actual": test_y.values, "predicted": preds}              # NEW
            ).to_csv(csv_path, index=False)                                # NEW

            fig, ax = plt.subplots()                                       # NEW
            ax.scatter(test_y, preds, alpha=0.5)                           # NEW
            lo, hi = float(test_y.min()), float(test_y.max())              # NEW
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)                # NEW
            ax.set_xlabel("actual"); ax.set_ylabel("predicted")            # NEW
            ax.set_title(f"alpha={req.alpha} | l1_ratio={req.l1_ratio}")   # NEW
            png_path = os.path.join(tmpdir, "predictions_plot.png")        # NEW
            fig.savefig(png_path, dpi=120, bbox_inches="tight")            # NEW
            plt.close(fig)                                                 # NEW

            mlflow.log_artifact(csv_path)                                  # NEW (single file)
            mlflow.log_artifact(png_path)                                  # NEW (single file)
            mlflow.log_artifacts(tmpdir, artifact_path="all_outputs")      # NEW (folder)

        artifact_uri = mlflow.get_artifact_uri()

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "params": {"alpha": req.alpha, "l1_ratio": req.l1_ratio},
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "artifact_uri": artifact_uri,
    }
```

> [!IMPORTANT]
> `mlflow.get_artifact_uri()` must be called **inside** the `with mlflow.start_run(...)` block, otherwise there is no active run and the call raises an exception.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Updated `streamlit/app/app.py` (show artifact URI)

```python
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")
MLFLOW_PUBLIC_URL = os.getenv("MLFLOW_PUBLIC_URL", "http://localhost:5000")

st.title("Chapter 10 — log_artifact / log_artifacts")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
l1_ratio = col2.number_input("l1_ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
run_name = st.text_input("Optional run name", value="")

if st.button("Train"):
    payload = {"alpha": alpha, "l1_ratio": l1_ratio}
    if run_name.strip():
        payload["run_name"] = run_name.strip()
    r = requests.post(f"{API_URL}/train", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    st.success(f"Run `{data['run_name']}` created in `{data['experiment']}`")

    m = data["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{m['rmse']:.4f}")
    c2.metric("MAE", f"{m['mae']:.4f}")
    c3.metric("R²", f"{m['r2']:.4f}")

    with st.expander("Artifact URI"):
        st.code(data["artifact_uri"])
        st.markdown(
            f"Open the run in MLflow UI: "
            f"[{MLFLOW_PUBLIC_URL}]({MLFLOW_PUBLIC_URL})"
        )
    st.json(data)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run the stack

```bash
cd chap10-saving-mlflow-artifacts-with-log-artifact-and-log-artifacts
docker compose up --build
```

Trigger one or two trainings:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.3,\"l1_ratio\":0.3}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.7,\"l1_ratio\":0.7}"
```

Or click **Train** in the Streamlit UI at [http://localhost:8501](http://localhost:8501).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Visualize artifacts in the MLflow UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap10`** → click on a run → tab **Artifacts**.

You should now see:

```text
artifacts/
├── all_outputs/                  ← logged by log_artifacts(tmpdir, artifact_path="all_outputs")
│   ├── predictions_plot.png
│   └── test_predictions.csv
├── predictions_plot.png          ← logged by log_artifact(png_path)
└── test_predictions.csv          ← logged by log_artifact(csv_path)
```

Click on `predictions_plot.png` — MLflow renders it inline in the browser. Click on `test_predictions.csv` — MLflow shows a preview of the first rows. Both are downloadable from a small button at the top right.

> [!NOTE]
> The `artifact_path="all_outputs"` argument creates a **subfolder** inside the run's artifact store. Useful when you log many things and want to keep them organized (`models/`, `plots/`, `data/`…).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Mini exercise — log a third artifact

Right before `mlflow.log_artifact(png_path)`, add a small text report and log it:

```python
report_path = os.path.join(tmpdir, "report.md")
with open(report_path, "w") as f:
    f.write(f"# Run report\n\n"
            f"- alpha = {req.alpha}\n"
            f"- l1_ratio = {req.l1_ratio}\n"
            f"- RMSE = {rmse:.4f}\n")
mlflow.log_artifact(report_path)
```

Restart the FastAPI container with `docker compose restart fastapi`, run a new training, and verify `report.md` shows up in the run's artifacts.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. `log_artifact` vs `log_artifacts` — when to use which

| | `log_artifact` (singular) | `log_artifacts` (plural) |
|---|---|---|
| Argument | A **file path** | A **directory path** |
| Behavior | Uploads that one file | Uploads everything inside (recursively) |
| Optional `artifact_path=` | Yes — places file under a subfolder in the run | Yes — same |
| Typical use | A single PNG, CSV, JSON, log | A `models/` or `data/` folder |

> [!IMPORTANT]
> If you pass a **directory** to `log_artifact` (singular), MLflow does **not** raise an error — it just logs an empty entry. Always double-check the trailing `s`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11b"></a>

## 12. Tear down

```bash
docker compose down
```

Volumes are kept again. The MLflow UI now contains 4 cumulative experiments (chap07 → chap10). Each chapter adds something visible.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 13. Recap

- An **artifact** is any file you want to keep with a run (CSV, PNG, model, report…).
- `mlflow.log_artifact(path)` attaches **one file**, `mlflow.log_artifacts(folder)` attaches a **whole folder**.
- Use `tempfile.TemporaryDirectory()` to build artifacts on the fly without leaving clutter inside the container.
- `mlflow.get_artifact_uri()` returns the storage URI — handy in API responses.
- The Docker stack is unchanged; we just added `matplotlib` to `fastapi/requirements.txt`.

> [!IMPORTANT]
> Next chapter (11) keeps the same stack and adds the next single concept: `mlflow.set_tag()` / `set_tags()` to attach metadata (engineering team, release version, environment) to each run.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 10 — <code>log_artifact()</code> + <code>log_artifacts()</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
