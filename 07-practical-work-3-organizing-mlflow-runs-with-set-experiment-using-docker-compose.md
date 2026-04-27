<a id="top"></a>

# Chapter 07 — Today's topic: `mlflow.set_experiment()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What is an MLflow experiment?](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The line we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Run the stack](#section-7) |
| 8 | [Visualize in the MLflow UI](#section-8) |
| 9 | [Mini exercise](#section-9) |
| 10 | [Tear down](#section-10) |
| 11 | [`set_experiment` vs `create_experiment`](#section-11) |
| 12 | [Recap](#section-12) |

---

<a id="section-1"></a>

## 1. Objective

Today we focus on **one MLflow function only**: `mlflow.set_experiment()`.

Goals:

- understand what an MLflow **experiment** is and why we need it;
- add **one new line** in the FastAPI service so that every training run is grouped under a named experiment;
- run the whole stack (MLflow + FastAPI + Streamlit) with `docker compose up`, trigger a training, and see the experiment appear in the MLflow UI;
- end the chapter cleanly with `docker compose down` before moving on.

> [!NOTE]
> The Docker / Compose plumbing is exactly the same as in **Chapter 06**. The only thing that changes from now on, chapter after chapter, is **one line of MLflow code** inside `fastapi/app/main.py`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What is an MLflow experiment?

An **experiment** in MLflow is a logical container that groups several **runs** together.

Think of it like a folder:

```text
Experiment "wine_quality_chap07"
├── Run #1 (alpha=0.5, l1_ratio=0.5)
├── Run #2 (alpha=0.7, l1_ratio=0.3)
└── Run #3 (alpha=0.1, l1_ratio=0.9)
```

Every time you call `mlflow.start_run()`, MLflow needs to know **inside which experiment** the run should be stored. If you do not set one explicitly, MLflow uses the experiment named `Default`, and over time everything ends up in the same big bag.

`mlflow.set_experiment("name")`:

- if the experiment exists → MLflow selects it for the next runs;
- if it does **not** exist yet → MLflow creates it on the fly.

> [!IMPORTANT]
> An experiment is the natural unit of organization in MLflow. Use one experiment per question you want to answer (one model, one dataset version, one business metric).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.set_tracking_uri(uri)` | Sets the URL of the MLflow tracking server (here: `http://mlflow:5000` inside the Docker network). |
| `mlflow.set_experiment(name)` | **NEW today.** Selects an experiment by name. Creates it if it does not exist. |
| `mlflow.start_run()` | Starts a new run inside the currently selected experiment (already used as basic plumbing). |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The line we add today

In `fastapi/app/main.py`, just one line is new compared to Chapter 06:

```python
mlflow.set_experiment("wine_quality_chap07")   # NEW
```

Everything else (the FastAPI app, the Dockerfiles, the `docker-compose.yml`) is identical to `chap06-mlops-stack/`.

> [!NOTE]
> Keep this discipline for every chapter from now on: **one new MLflow line per chapter**. That way you always know which line is responsible for the change you observe in the UI.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap07-organizing-mlflow-runs-with-set-experiment/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py                ← the only file that changes today
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py
```

The folder is a **fresh, self-contained project**: you can `cd` into it and run `docker compose up` without touching anything else.

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

EXPERIMENT_NAME = "wine_quality_chap07"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 07 — mlflow.set_experiment()")


def load_data() -> pd.DataFrame:
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep=";")


class TrainRequest(BaseModel):
    alpha: float = 0.5
    l1_ratio: float = 0.5


@app.get("/")
def root():
    return {"chapter": "07", "topic": "set_experiment", "mlflow": MLFLOW_URI}


@app.post("/train")
def train(req: TrainRequest):
    data = load_data()
    train_df, _ = train_test_split(data, random_state=40)
    train_x = train_df.drop(["quality"], axis=1)
    train_y = train_df["quality"]

    mlflow.set_experiment(EXPERIMENT_NAME)   # NEW

    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        return {
            "experiment": EXPERIMENT_NAME,
            "run_id": run.info.run_id,
            "alpha": req.alpha,
            "l1_ratio": req.l1_ratio,
        }
```

> [!IMPORTANT]
> The run created here is **empty** on purpose: we have not called `log_param`, `log_metric`, or `log_artifact` yet. Today's only job is to verify that the **experiment** appears in the MLflow UI.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Run the stack

From the chapter folder:

```bash
cd chap07-organizing-mlflow-runs-with-set-experiment
docker compose up --build
```

Wait until the three services are up. Then trigger a training from the Streamlit UI or directly with `curl`:

```bash
curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
```

The API answers with the experiment name and the new `run_id`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Visualize in the MLflow UI

Open [http://localhost:5000](http://localhost:5000).

You should observe:

1. a **new experiment** in the left sidebar named `wine_quality_chap07`;
2. inside that experiment, **one run** with an auto-generated name (e.g. `selective-dog-127`);
3. the run is **almost empty** — no params, no metrics — and that is correct for today's chapter.

> [!NOTE]
> Trigger a few more `POST /train` calls. Each call creates a new run **inside the same experiment**, which is exactly what `set_experiment` is for: grouping related runs together.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Mini exercise

In `fastapi/app/main.py`, change the experiment name once and rebuild:

```python
EXPERIMENT_NAME = "wine_quality_chap07_v2"   # try a second name
```

Then:

```bash
docker compose up -d --build fastapi
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{}"
```

Refresh the MLflow UI. You should now see **two experiments** in the left sidebar:

- `wine_quality_chap07` (3 runs from the previous step)
- `wine_quality_chap07_v2` (1 run from the new call)

> [!IMPORTANT]
> This proves that `set_experiment` always points to the **right container** for the next runs, and that switching name is enough to start a new logical group.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Tear down

Always finish a chapter cleanly before opening the next one:

```bash
docker compose down
```

> [!WARNING]
> Do **not** use `docker compose down -v` here. We want to keep the MLflow data (volumes) so we can compare what each chapter adds in the UI.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. `set_experiment` vs `create_experiment`

There is a second function, `mlflow.create_experiment`, that does almost the same job. The difference is important.

| Function | Behavior if the experiment already exists | Returns | Typical use |
|---|---|---|---|
| `mlflow.set_experiment(name)` | re-uses it silently | the `Experiment` object | day-to-day code (this chapter) |
| `mlflow.create_experiment(name, artifact_location=..., tags=...)` | raises an error | the new `experiment_id` | one-shot creation when you need control over the artifact location or tags |

> [!NOTE]
> `set_experiment` is the right default in production code. It is **idempotent**: running the same script twice never crashes because the experiment already exists.

A `create_experiment` example, for reference only:

```python
exp_id = mlflow.create_experiment(
    name="wine_quality_chap07_explicit",
    artifact_location="file:/mlflow/mlruns",
    tags={"chapter": "07", "owner": "you"},
)
with mlflow.start_run(experiment_id=exp_id):
    ...
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Recap

- An MLflow **experiment** is a named container for runs.
- `mlflow.set_experiment("name")` selects it (and creates it if missing).
- One new line in `fastapi/app/main.py` is enough to organize every future run under that name.
- The Docker stack is unchanged compared to Chapter 06 — and that will stay true for the next chapters too.

> [!IMPORTANT]
> Tomorrow (Chapter 08), we keep the same project skeleton and add the **next single line**: `mlflow.start_run(run_name=...)` to give human-readable names to runs.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 07 — <code>mlflow.set_experiment()</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
