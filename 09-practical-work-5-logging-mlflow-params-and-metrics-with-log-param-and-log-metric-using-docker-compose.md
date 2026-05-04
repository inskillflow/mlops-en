<a id="top"></a>

# Chapter 09 — Today's topic: `mlflow.log_param()` + `mlflow.log_metric()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Param vs Metric — what is the difference?](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The lines we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Run the stack](#section-7) |
| 8 | [Visualize in the MLflow UI](#section-8) |
| 9 | [Mini exercise — compare 5 runs](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [Single-call vs batch helpers](#section-11) |
| 12 | [Recap](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack as Chapters 07 and 08 and add **five new MLflow lines**:

- `mlflow.log_param("alpha", ...)` and `mlflow.log_param("l1_ratio", ...)` — record the **hyperparameters**;
- `mlflow.log_metric("rmse", ...)`, `mlflow.log_metric("mae", ...)`, `mlflow.log_metric("r2", ...)` — record the **evaluation metrics**.

After this chapter, the runs in the MLflow UI finally **stop being empty**. Each run shows its 2 params and its 3 metrics, and you can compare runs side by side.

> [!IMPORTANT]
> Logging params and metrics is the **first real value** of MLflow. Up to now, we just had named runs. Now we have **measurable, comparable, reproducible** runs.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Param vs Metric — what is the difference?

| Concept | When | Mutability | Examples |
|---|---|---|---|
| **Param** (`log_param`) | **Before** training (input of the experiment) | Logged once, never changes | `alpha`, `l1_ratio`, `learning_rate`, `n_estimators` |
| **Metric** (`log_metric`) | **After** (or during) training (output of the experiment) | Can be logged many times (one value per step) | `rmse`, `mae`, `r2`, `accuracy`, `loss_at_epoch_42` |

Mental rule:

- if it goes **into** the training, it is a **param**;
- if it comes **out** of the training, it is a **metric**.

> [!NOTE]
> Metrics support a `step=` argument. That is what produces those nice line charts in the MLflow UI when you log the same metric at every epoch (`mlflow.log_metric("loss", value, step=epoch)`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.log_param(key, value)` | **NEW today.** Records one hyperparameter. |
| `mlflow.log_metric(key, value, step=None)` | **NEW today.** Records one evaluation metric. |
| `mlflow.log_params({...})` | (Bonus) Logs a whole dictionary of params at once. |
| `mlflow.log_metrics({...}, step=None)` | (Bonus) Logs a whole dictionary of metrics at once. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The lines we add today

Five new lines in `fastapi/app/main.py`, all inside the `with mlflow.start_run(...)` block:

```python
mlflow.log_param("alpha", req.alpha)         # NEW
mlflow.log_param("l1_ratio", req.l1_ratio)   # NEW
mlflow.log_metric("rmse", rmse)              # NEW
mlflow.log_metric("mae", mae)                # NEW
mlflow.log_metric("r2", r2)                  # NEW
```

We also add a tiny helper `eval_metrics(actual, pred)` that returns `(rmse, mae, r2)` so that the logging stays clean.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap09-logging-params-and-metrics-with-log-param-and-log-metric/
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
        └── app.py             ← shows the 3 metrics in styled tiles
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Modified code — `fastapi/app/main.py`

```python
import io
import os

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

EXPERIMENT_NAME = "wine_quality_chap09"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 09 — log_param / log_metric")


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

        mlflow.log_param("alpha", req.alpha)        # NEW
        mlflow.log_param("l1_ratio", req.l1_ratio)  # NEW
        mlflow.log_metric("rmse", rmse)             # NEW
        mlflow.log_metric("mae", mae)               # NEW
        mlflow.log_metric("r2", r2)                 # NEW

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "params": {"alpha": req.alpha, "l1_ratio": req.l1_ratio},
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
    }
```

> [!NOTE]
> Notice that the metrics computation (`eval_metrics`) sits **inside** the `with mlflow.start_run()` block. That way, if the metrics computation fails, MLflow still closes the run cleanly with status `FAILED`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run the stack

```bash
cd chap09-logging-params-and-metrics-with-log-param-and-log-metric
docker compose up --build
```

Trigger five trainings with different hyperparameters:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.1,\"l1_ratio\":0.1}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.3,\"l1_ratio\":0.3}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.7,\"l1_ratio\":0.7}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.9,\"l1_ratio\":0.9}"
```

Or just click **Train** five times in the Streamlit UI at [http://localhost:8501](http://localhost:8501) with different values.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Visualize in the MLflow UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap09`**.

You should now observe:

1. each run shows **2 params** (`alpha`, `l1_ratio`) and **3 metrics** (`rmse`, `mae`, `r2`) in the table view;
2. the table can be **sorted** by any metric column (click the column header) — try sorting by `rmse` ascending to find the best run;
3. selecting two or more runs and clicking **Compare** opens a side-by-side view with a **parallel coordinates plot** for params and metrics.

> [!IMPORTANT]
> The whole point of MLflow Tracking shows up here for the first time: you can **rank and compare** runs based on objective numbers, not on the order in which you ran them.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Mini exercise — compare 5 runs

In the MLflow UI:

1. select the 5 runs you just created;
2. click **Compare**;
3. find the run with the **lowest `rmse`**;
4. note its `alpha` and `l1_ratio` values.

Now ask yourself: would you have found that combination as quickly with **just `print()` in the terminal**? That is the practical value MLflow brings.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Tear down

```bash
docker compose down
```

> [!NOTE]
> We keep the volumes again. Over Chapters 07 → 09, the MLflow UI now contains three different experiments side by side, which is exactly what we want for comparison.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Single-call vs batch helpers

For two params and three metrics, the single-call version is fine. As soon as you have 10+ params, prefer the batch helpers:

```python
mlflow.log_params({
    "alpha": req.alpha,
    "l1_ratio": req.l1_ratio,
    "random_state": 42,
})

mlflow.log_metrics({
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
})
```

> [!IMPORTANT]
> `log_param` is **idempotent only the first time**. Calling `log_param("alpha", 0.5)` then `log_param("alpha", 0.7)` in the same run raises an error. Metrics, on the contrary, can be logged many times (with different `step` values) — that is how training curves are built.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap

- `log_param` records the **inputs** of a run (hyperparameters).
- `log_metric` records the **outputs** (evaluation metrics, training curves).
- After this chapter, runs are **comparable**: ranking, sorting, parallel-coordinates plots all work in the MLflow UI.
- The Docker stack is unchanged again. Only **five MLflow lines** changed.

> [!IMPORTANT]
> Next chapter (10) keeps the same stack and adds the next single line: `mlflow.log_artifact()` to attach files (a CSV, a PNG plot) to each run.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 09 — <code>log_param()</code> + <code>log_metric()</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
