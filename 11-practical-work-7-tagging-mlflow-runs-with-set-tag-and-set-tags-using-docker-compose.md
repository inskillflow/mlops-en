<a id="top"></a>

# Chapter 11 — Today's topic: `mlflow.set_tag()` + `mlflow.set_tags()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Tag vs Param vs Metric](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The lines we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Updated `streamlit/app/app.py` (let user pass tags)](#section-7) |
| 8 | [Run the stack](#section-8) |
| 9 | [Visualize tags in the MLflow UI](#section-9) |
| 10 | [Filter runs by tag — the **real** value of tags](#section-10) |
| 11 | [Mini exercise — split runs by team](#section-11) |
| 12 | [Tear down](#section-12) |
| 13 | [Recap](#section-13) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack and we add **one new MLflow line** (with two flavors):

- `mlflow.set_tag(key, value)` — attaches a single key/value **metadata** to the run;
- `mlflow.set_tags({...})` — attaches **a whole dictionary** of tags in one shot.

Tags are the answer to questions like:
- *"Who launched this run?"* — `engineering = "ML platform"`
- *"What is the release stage?"* — `release.stage = "RC1"`
- *"Which Git branch?"* — `git.branch = "feature/elasticnet-tuning"`
- *"What is the data source?"* — `dataset = "winequality-red-v2"`

> [!IMPORTANT]
> Tags are **searchable in the MLflow UI** (in the search bar: `tags.engineering = "ML platform"`). That's the whole point — being able to slice your run history by team, branch, environment, dataset version… anything textual.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Tag vs Param vs Metric

Three concepts that look similar but answer different questions:

| Concept | Question it answers | Type | Mutable? |
|---|---|---|---|
| **Param** | What hyperparameter went *into* the training? | Numeric or string | No (logged once) |
| **Metric** | What number came *out* of the training? | Numeric only | Yes (one value per step) |
| **Tag** | Metadata about *who/what/where/why* | String only | **Yes** (can be overwritten) |

> [!NOTE]
> A tag can be **updated** for a run, even after the run is finished — using `MlflowClient().set_tag(run_id, key, value)`. This is unique to tags. We'll use this property in a later chapter when we set the tag `model.stage = "production"` after a manual review.

A few **system tags** (set by MLflow automatically) are particularly useful:

| Auto tag | Set when |
|---|---|
| `mlflow.user` | The OS user that ran the script |
| `mlflow.source.name` | The script path |
| `mlflow.source.git.commit` | If the working dir is a git repo |
| `mlflow.runName` | The `run_name=` you passed to `start_run` (Chapter 08) |

In a Docker container, `mlflow.user` will typically be `root`, and `mlflow.source.git.commit` is empty unless you copy the `.git` folder into the image. **Custom tags are how you fix that** — log `git.commit` yourself from the host environment.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.set_tag(key, value)` | **NEW today.** Attach one metadata pair. |
| `mlflow.set_tags({...})` | **NEW today.** Attach a whole dict in one call. |
| `mlflow.delete_tag(key)` | (Bonus) Remove a tag from the active run. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The lines we add today

Inside the `with mlflow.start_run(...)` block, **right after** `start_run` (so tags appear from the very first record):

```python
mlflow.set_tags({                                # NEW
    "engineering": "ML platform",
    "release.candidate": "RC1",
    "release.version": "2.0",
    "model.family": "elasticnet",
    "dataset": "winequality-red",
})

mlflow.set_tag("triggered_by", req.triggered_by)  # NEW (single tag, dynamic)
```

Note we mix:
- **static tags** (the same for every run — team, dataset, model family) → grouped in `set_tags({...})`;
- **dynamic tags** (vary per request — `triggered_by` = "streamlit-ui" or "curl-cli") → one call to `set_tag()`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap11-tagging-mlflow-runs-with-set-tag-and-set-tags/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py            ← changes today
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py             ← shows the tags section
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
matplotlib.use("Agg")
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

EXPERIMENT_NAME = "wine_quality_chap11"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

STATIC_TAGS = {                                 # NEW
    "engineering": "ML platform",
    "release.candidate": "RC1",
    "release.version": "2.0",
    "model.family": "elasticnet",
    "dataset": "winequality-red",
}

app = FastAPI(title="Chapter 11 — set_tag / set_tags")


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
    triggered_by: str = "curl-cli"   # NEW: free-form string we will tag


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
        mlflow.set_tags(STATIC_TAGS)                       # NEW (bulk)
        mlflow.set_tag("triggered_by", req.triggered_by)   # NEW (single)

        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        preds = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)

        mlflow.log_param("alpha", req.alpha)
        mlflow.log_param("l1_ratio", req.l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_predictions.csv")
            pd.DataFrame(
                {"actual": test_y.values, "predicted": preds}
            ).to_csv(csv_path, index=False)

            fig, ax = plt.subplots()
            ax.scatter(test_y, preds, alpha=0.5)
            lo, hi = float(test_y.min()), float(test_y.max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            ax.set_xlabel("actual"); ax.set_ylabel("predicted")
            ax.set_title(f"alpha={req.alpha} | l1_ratio={req.l1_ratio}")
            png_path = os.path.join(tmpdir, "predictions_plot.png")
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(png_path)

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "params": {"alpha": req.alpha, "l1_ratio": req.l1_ratio},
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "tags": {**STATIC_TAGS, "triggered_by": req.triggered_by},
    }
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Updated `streamlit/app/app.py` (let user pass tags)

```python
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("Chapter 11 — set_tag / set_tags")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", 0.0, 1.0, 0.5, 0.1)
l1_ratio = col2.number_input("l1_ratio", 0.0, 1.0, 0.5, 0.1)
run_name = st.text_input("Run name (optional)")
triggered_by = st.selectbox(
    "triggered_by",
    options=["streamlit-ui", "curl-cli", "scheduled-job", "data-scientist-A"],
)

if st.button("Train", type="primary"):
    payload = {"alpha": alpha, "l1_ratio": l1_ratio, "triggered_by": triggered_by}
    if run_name.strip():
        payload["run_name"] = run_name.strip()
    r = requests.post(f"{API_URL}/train", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    st.success(f"Run `{data['run_name']}` created in `{data['experiment']}`")

    m = data["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{m['rmse']:.4f}")
    c2.metric("MAE", f"{m['mae']:.4f}")
    c3.metric("R²", f"{m['r2']:.4f}")

    with st.expander("Tags attached to this run"):
        st.json(data["tags"])
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run the stack

```bash
cd chap11-tagging-mlflow-runs-with-set-tag-and-set-tags
docker compose up --build
```

Trigger a few runs with different `triggered_by`:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.3,\"triggered_by\":\"streamlit-ui\"}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"triggered_by\":\"curl-cli\"}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.7,\"triggered_by\":\"scheduled-job\"}"
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Visualize tags in the MLflow UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap11`** → click any run.

In the right pane you should now see a section **Tags** with rows like:

```text
engineering         ML platform
release.candidate   RC1
release.version     2.0
model.family        elasticnet
dataset             winequality-red
triggered_by        streamlit-ui
```

In the **runs table view**, click the **Columns** dropdown (top-left of the table) and tick the tag columns you want to display. They appear as plain columns next to params and metrics — same UX, but searchable.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Filter runs by tag — the **real** value of tags

In the MLflow UI search bar (above the runs table), type:

```text
tags.triggered_by = "scheduled-job"
```

Press Enter — only runs that came from the scheduled job remain. Combine filters:

```text
tags.triggered_by = "streamlit-ui" and metrics.rmse < 0.7
```

Now you see only the **good runs** that were **launched from the UI**. This is how you make sense of hundreds of runs in a real project.

> [!IMPORTANT]
> The same syntax works in the Python API: `MlflowClient().search_runs(experiment_ids=[id], filter_string='tags.triggered_by = "streamlit-ui"')`. We'll use it later for automated dashboards.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Mini exercise — split runs by team

1. Send 3 trainings with `triggered_by="data-scientist-A"`.
2. Send 3 trainings with `triggered_by="data-scientist-B"`.
3. In the UI, search `tags.triggered_by = "data-scientist-A"` → should show 3 runs.
4. Sort the result by `metrics.rmse` and identify A's best run.
5. Same for B. Compare.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Tear down

```bash
docker compose down
```

> [!NOTE]
> Tags are stored in the **backend store** (`sqlite:///database/mlflow.db`), not in the artifact store. They survive `docker compose down` as long as the `mlflow-db` volume is kept.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-13"></a>

## 13. Recap

- **Tags** = string metadata attached to a run (team, branch, dataset version, environment, who triggered…).
- `mlflow.set_tag(key, value)` for one tag, `mlflow.set_tags({...})` for many.
- Tags are **mutable** (they can be updated post-run via `MlflowClient`), which makes them perfect for lifecycle markers (`status = "approved"`).
- The whole point is **filtering**: `tags.<key> = "<value>"` in the UI search bar.
- The Docker stack is unchanged.

> [!IMPORTANT]
> Next chapter (12) keeps the same stack and adds the next pattern: **launching multiple runs and multiple experiments** in a single API call (a `for` loop with `mlflow.start_run` per iteration), which is how grid searches are recorded.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 11 — <code>set_tag()</code> + <code>set_tags()</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
