<a id="top"></a>

# Chapter 08 — Today's topic: `start_run(run_name=...)` + `last_active_run()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What is a run? Why give it a name?](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The lines we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Run the stack](#section-7) |
| 8 | [Visualize in the MLflow UI](#section-8) |
| 9 | [Mini exercise](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [`active_run` vs `last_active_run`](#section-11) |
| 12 | [Recap](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack as Chapter 07 and add **two new MLflow lines**:

- `mlflow.start_run(run_name="...")` — give the run a **human-readable name** instead of relying on the auto-generated one (`selective-dog-127`);
- `mlflow.last_active_run()` — read information about the run **after the `with` block has closed**, for example to return the `run_id` to the API caller.

> [!IMPORTANT]
> Naming runs is one of the cheapest yet most useful habits in MLflow. With dozens or hundreds of runs in an experiment, animal-style names become unmanageable.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What is a run? Why give it a name?

A **run** is one execution of your training code, with a specific set of parameters and a specific outcome.

By default, MLflow names runs with random animal+adjective pairs:

```text
selective-dog-127
thoughtful-mare-378
crawling-goose-215
```

Cute, but useless when you want to find *"the run with alpha=0.7 from yesterday"*. With `run_name=`, **you** decide:

```text
baseline_v1
alpha_0.7_l1_0.3
prod_candidate_2026_04_27
```

The function `mlflow.last_active_run()` is the second piece. It returns the `Run` object of the **last run that just finished** — useful when the `with mlflow.start_run():` block has already closed but you still want to read its `run_id`, `run_name`, or `status` (typically to return it from a REST endpoint).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.start_run(run_name="...")` | **NEW today.** Starts a run with an explicit name. |
| `mlflow.last_active_run()` | **NEW today.** Returns the last finished run as a `Run` object. |
| `mlflow.active_run()` | (For comparison) Returns the **currently open** run, or `None` if no run is open. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The lines we add today

Two new lines in `fastapi/app/main.py`:

```python
with mlflow.start_run(run_name=run_name):    # NEW: run_name
    ...

last = mlflow.last_active_run()              # NEW: query the run after it ended
```

The body of the `with` block is unchanged.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

Same skeleton as Chapter 07 — only `fastapi/app/main.py` and `streamlit/app/app.py` change.

```text
chap08-naming-mlflow-runs-with-start-run-and-last-active-run/
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
        └── app.py             ← adds a "Run name" text input
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Modified code — `fastapi/app/main.py`

```python
import io
import os

import mlflow
import pandas as pd
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap08"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 08 — start_run / last_active_run")


def load_data() -> pd.DataFrame:
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep=";")


class TrainRequest(BaseModel):
    alpha: float = 0.5
    l1_ratio: float = 0.5
    run_name: str | None = None


@app.post("/train")
def train(req: TrainRequest):
    data = load_data()
    train_df, _ = train_test_split(data, random_state=40)
    train_x = train_df.drop(["quality"], axis=1)
    train_y = train_df["quality"]

    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = req.run_name or f"alpha_{req.alpha}_l1_{req.l1_ratio}"

    with mlflow.start_run(run_name=run_name):                       # NEW
        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

    last = mlflow.last_active_run()                                 # NEW
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "status": last.info.status,
        "alpha": req.alpha,
        "l1_ratio": req.l1_ratio,
    }
```

> [!NOTE]
> If the caller does not provide a `run_name`, we fall back to a deterministic name built from the hyperparameters. That alone is already much more useful than `selective-dog-127`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run the stack

```bash
cd chap08-naming-mlflow-runs-with-start-run-and-last-active-run
docker compose up --build
```

Trigger two trainings with explicit names:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" \
     -d "{\"alpha\":0.5,\"l1_ratio\":0.5,\"run_name\":\"baseline_v1\"}"

curl -X POST http://localhost:8000/train -H "Content-Type: application/json" \
     -d "{\"alpha\":0.7,\"l1_ratio\":0.3,\"run_name\":\"baseline_v2\"}"
```

The API now returns the `run_id`, the `run_name` you asked for, and the final `status` (`FINISHED` if all went well).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Visualize in the MLflow UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap08`**.

You should see two runs literally called `baseline_v1` and `baseline_v2` in the **Run name** column — no more random animals.

> [!IMPORTANT]
> The `run_name` is a **mutable** label, but the `run_id` is the **immutable** identifier. Always use the `run_id` for programmatic references; use `run_name` only for humans reading the UI.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Mini exercise

1. Trigger a training **without** providing `run_name`. Observe in the UI: the run is named `alpha_0.5_l1_0.5` (the deterministic fallback we wrote).
2. Trigger a second training with the **same** hyperparameters. MLflow does **not** complain — two runs can share the same `run_name`. Only the `run_id` is unique.
3. Inside the `with` block, add a temporary `print(mlflow.active_run().info.run_id)` and rebuild. Notice that `active_run()` returns `None` **after** the `with` block closes, while `last_active_run()` keeps returning the last finished run.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Tear down

```bash
docker compose down
```

> [!NOTE]
> We keep the volumes on purpose: the runs from this chapter remain visible in the MLflow UI after restart, side by side with the runs from Chapter 07.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. `active_run` vs `last_active_run`

| Function | Returns | Typical use |
|---|---|---|
| `mlflow.active_run()` | The run that is currently **open**, or `None` outside a `with` block | Inside the `with` block when you want to access the current `run_id` |
| `mlflow.last_active_run()` | The run that **just finished**, even after the `with` block closed | Right after the `with` block when you want to return the `run_id` to the caller |

> [!WARNING]
> Calling `mlflow.active_run().info.run_id` **outside** the `with` block raises `AttributeError: 'NoneType' object has no attribute 'info'`. That is exactly the case where you must use `last_active_run()` instead.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap

- `start_run(run_name=...)` makes runs **searchable by humans** in the UI.
- `last_active_run()` lets you **read the run after the `with` block** has closed.
- The Docker stack is identical to Chapter 07. Only **two MLflow lines** changed.

> [!IMPORTANT]
> Next chapter (09) keeps the same stack again and adds the **next two MLflow lines**: `mlflow.log_param()` and `mlflow.log_metric()` — so the runs finally stop being empty.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 08 — <code>start_run(run_name=...)</code> + <code>last_active_run()</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
