<a id="top"></a>

# Chapter 17 — Today's topic: Model Registry — `register_model` + `MlflowClient`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Why a Model Registry?](#section-2) |
| 3 | [Tracking vs Registry — the mental model](#section-3) |
| 4 | [The 4 lifecycle stages](#section-4) |
| 5 | [MLflow functions introduced](#section-5) |
| 6 | [The lines we add today](#section-6) |
| 7 | [Project structure](#section-7) |
| 8 | [Modified code — `fastapi/app/main.py`](#section-8) |
| 9 | [Updated `streamlit/app/app.py`](#section-9) |
| 10 | [Run the stack](#section-10) |
| 11 | [End-to-end demo: train → register → promote → consume](#section-11) |
| 12 | [Visualize the Registry in the UI](#section-12) |
| 13 | [Tear down](#section-13) |
| 14 | [Recap](#section-14) |

---

<a id="section-1"></a>

## 1. Objective

Up to chapter 16 every model lives **inside its own run** (URI `runs:/<run_id>/<artifact_path>`). That's enough to load it back, but consumers must know the random run id. Today we add the **Model Registry**: a named, versioned, governed place where models live independently from the runs that produced them.

We will add three FastAPI endpoints:

- `POST /train-and-register` — train + log + **`mlflow.register_model(...)`** under the name `WineQualityPredictor`;
- `POST /promote` — use **`MlflowClient.transition_model_version_stage(...)`** to move a version to `Staging`, `Production`, or `Archived`;
- `POST /predict-production` — load **`models:/WineQualityPredictor/Production`** and predict (no `run_id` needed by consumers).

> [!IMPORTANT]
> The Registry is what turns a research artifact into a production-ready, governed model. It's the single most useful MLflow feature for **MLOps**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Why a Model Registry?

Without a Registry, your downstream service has to know:

```text
"runs:/8a4f2e91d7b14d8b91d6c0e1ad1ce4be/wine_quality_pyfunc"
```

That's brittle. With the Registry, the same service uses:

```text
"models:/WineQualityPredictor/Production"
```

A human (or a CI job) decides which **version** is in `Production`. The downstream code never changes. This decoupling is the whole point.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Tracking vs Registry — the mental model

```text
                ┌──────────────────────────┐
   Tracking →   │  Experiments → Runs       │     (every train(), every metric, all artifacts)
                │  Run 8a4f...  ▸ pyfunc    │
                │  Run d3e1...  ▸ pyfunc    │
                │  Run 92c0...  ▸ pyfunc    │
                └────────────┬─────────────┘
                             │ register_model(...)
                             ▼
                ┌──────────────────────────┐
   Registry →   │  WineQualityPredictor    │     (named, versioned, with stages)
                │  ├ v1   stage: Archived   │
                │  ├ v2   stage: Staging    │
                │  └ v3   stage: Production │
                └──────────────────────────┘
                             │ load_model("models:/.../Production")
                             ▼
                ┌──────────────────────────┐
   Consumer →   │  FastAPI / Streamlit /   │
                │  Spark / batch job ...    │
                └──────────────────────────┘
```

The same MLflow server hosts both. When you call `mlflow.register_model(uri, name)`, MLflow creates the registered model the first time and adds a new **version** afterwards.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The 4 lifecycle stages

| Stage | Meaning |
|---|---|
| `None` | Just registered, no decision yet. |
| `Staging` | Approved for shadow/QA traffic. |
| `Production` | Active, serves real traffic. |
| `Archived` | Retired — kept for audit/rollback. |

A model can have **many** versions but at most one in `Production` (and one in `Staging`) **at a time** (when you set `archive_existing_versions=True`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.register_model(model_uri, name)` | **NEW today.** Register a model URI under a name; create a new version. Returns a `ModelVersion`. |
| `mlflow.MlflowClient()` | **NEW today.** Lower-level client for Registry operations. |
| `client.transition_model_version_stage(name, version, stage, archive_existing_versions=...)` | **NEW today.** Move a version between stages. |
| `client.search_model_versions("name='X'")` | **NEW today.** List all versions of a registered model. |
| `mlflow.pyfunc.load_model("models:/<name>/<stage_or_version>")` | **NEW today.** Resolve & load directly from the Registry. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. The lines we add today

Right after `mlflow.pyfunc.log_model(...)` in the run:

```python
model_uri = f"runs:/{mlflow.active_run().info.run_id}/{ARTIFACT_PATH}"
mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)   # NEW
```

A new module-level client used by the new endpoints:

```python
client = mlflow.MlflowClient()                                                # NEW
```

The `/promote` endpoint:

```python
client.transition_model_version_stage(                                        # NEW
    name=REGISTERED_MODEL_NAME,
    version=req.version,
    stage=req.stage,
    archive_existing_versions=req.archive_existing,
)
```

The `/predict-production` endpoint:

```python
loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")   # NEW
```

That's it.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Project structure

```text
chap17-versioning-mlflow-models-with-the-model-registry-and-mlflowclient/
├── docker-compose.yml
├── mlflow/Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py            ← /train-and-register, /promote, /predict-production, /versions
│       └── wrapper.py
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/app.py             ← 3 tabs: Train+Register / Promote / Predict (Production)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Modified code — `fastapi/app/main.py`

```python
import io
import os
import tempfile

import cloudpickle
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
import sklearn
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.wrapper import WineQualityWrapper

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap17_registry"
ARTIFACT_PATH = "wine_quality_pyfunc"
REGISTERED_MODEL_NAME = "WineQualityPredictor"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 17 - Model Registry")
client = MlflowClient()                                            # NEW


def load_data() -> pd.DataFrame:
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep=";")


def eval_metrics(actual, pred):
    return (
        float(np.sqrt(mean_squared_error(actual, pred))),
        float(mean_absolute_error(actual, pred)),
        float(r2_score(actual, pred)),
    )


def build_conda_env() -> dict:
    return {
        "channels": ["defaults"],
        "dependencies": [
            "python=3.12", "pip",
            {"pip": [
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                "joblib", "numpy", "pandas",
            ]},
        ],
        "name": "wine_quality_env",
    }


class TrainRequest(BaseModel):
    alpha: float = 0.4
    l1_ratio: float = 0.4
    run_name: str | None = None


class PromoteRequest(BaseModel):
    version: int
    stage: str = "Production"          # None | Staging | Production | Archived
    archive_existing: bool = True


class PredictRequest(BaseModel):
    rows: list[dict]


@app.post("/train-and-register")
def train_and_register(req: TrainRequest):
    data = load_data()
    train_df, test_df = train_test_split(data, random_state=40)
    train_x = train_df.drop(["quality"], axis=1)
    test_x = test_df.drop(["quality"], axis=1)
    train_y = train_df["quality"]
    test_y = test_df["quality"]

    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = req.run_name or f"a{req.alpha}_l{req.l1_ratio}"

    with mlflow.start_run(run_name=run_name):
        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        preds = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)
        mlflow.log_params({"alpha": req.alpha, "l1_ratio": req.l1_ratio})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        signature = infer_signature(test_x, preds)
        with tempfile.TemporaryDirectory() as tmpdir:
            sk_path = os.path.join(tmpdir, "sklearn_model.pkl")
            joblib.dump(lr, sk_path)
            mlflow.pyfunc.log_model(
                artifact_path=ARTIFACT_PATH,
                python_model=WineQualityWrapper(),
                artifacts={"sklearn_model": sk_path},
                conda_env=build_conda_env(),
                signature=signature,
                input_example=test_x.head(5),
            )

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)   # NEW

    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": run_id,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "registered": {
            "name": mv.name,
            "version": int(mv.version),
            "current_stage": mv.current_stage,
            "source": mv.source,
        },
    }


@app.post("/promote")
def promote(req: PromoteRequest):
    if req.stage not in {"None", "Staging", "Production", "Archived"}:
        raise HTTPException(400, f"Invalid stage: {req.stage}")
    mv = client.transition_model_version_stage(                                       # NEW
        name=REGISTERED_MODEL_NAME,
        version=req.version,
        stage=req.stage,
        archive_existing_versions=req.archive_existing,
    )
    return {
        "name": mv.name,
        "version": int(mv.version),
        "new_stage": mv.current_stage,
        "archive_existing": req.archive_existing,
    }


@app.get("/versions")
def versions():
    items = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")           # NEW
    return [
        {
            "name": v.name,
            "version": int(v.version),
            "stage": v.current_stage,
            "run_id": v.run_id,
        }
        for v in items
    ]


@app.post("/predict-production")
def predict_production(req: PredictRequest):
    try:
        uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
        loaded = mlflow.pyfunc.load_model(uri)                                         # NEW
    except Exception as e:
        raise HTTPException(404, f"No Production model: {e}")
    df = pd.DataFrame(req.rows)
    preds = loaded.predict(df)
    return {"model_uri": uri, "predictions": list(preds)}
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Updated `streamlit/app/app.py`

```python
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
    a = st.number_input("alpha", 0.0, 1.0, 0.4, 0.1)
    l = st.number_input("l1_ratio", 0.0, 1.0, 0.4, 0.1)
    if st.button("Train + Register", type="primary"):
        r = requests.post(f"{API_URL}/train-and-register",
                          json={"alpha": a, "l1_ratio": l}, timeout=300)
        r.raise_for_status()
        st.json(r.json())

with tab_promote:
    if st.button("Refresh versions"):
        st.session_state["versions"] = requests.get(f"{API_URL}/versions").json()
    st.json(st.session_state.get("versions", []))
    v = st.number_input("Version", 1, 999, 1, 1)
    s = st.selectbox("Stage", ["Staging", "Production", "Archived", "None"])
    archive = st.checkbox("Archive existing in same stage", value=True)
    if st.button("Promote"):
        r = requests.post(f"{API_URL}/promote",
                          json={"version": v, "stage": s,
                                "archive_existing": archive}, timeout=60)
        r.raise_for_status()
        st.success(f"Version {v} → {s}")
        st.json(r.json())

with tab_predict:
    sample = {
        "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4,
    }
    payload_text = st.text_area(
        "JSON rows",
        value=json.dumps([sample], indent=2),
        height=280,
    )
    if st.button("Predict via Production"):
        try:
            r = requests.post(f"{API_URL}/predict-production",
                              json={"rows": json.loads(payload_text)}, timeout=120)
            r.raise_for_status()
            st.json(r.json())
        except requests.HTTPError as e:
            st.error(f"{e} — did you promote a version to Production?")
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Run the stack

```bash
cd chap17-versioning-mlflow-models-with-the-model-registry-and-mlflowclient
docker compose up --build
```

| Service | URL |
|---|---|
| MLflow UI (Experiments **+ Models** tabs) | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. End-to-end demo: train → register → promote → consume

```bash
# 1) Create v1
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" \
  -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"

# 2) Create v2 with different hyperparams
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" \
  -d "{\"alpha\":0.7,\"l1_ratio\":0.3}"

# 3) List versions
curl http://localhost:8000/versions

# 4) Promote v2 → Production (auto-archives v1 if v1 was already there)
curl -X POST http://localhost:8000/promote \
  -H "Content-Type: application/json" \
  -d "{\"version\":2,\"stage\":\"Production\",\"archive_existing\":true}"

# 5) Predict via the alias models:/WineQualityPredictor/Production
curl -X POST http://localhost:8000/predict-production \
  -H "Content-Type: application/json" \
  -d "{\"rows\":[{\"fixed acidity\":7.4,\"volatile acidity\":0.7,\"citric acid\":0.0,\"residual sugar\":1.9,\"chlorides\":0.076,\"free sulfur dioxide\":11.0,\"total sulfur dioxide\":34.0,\"density\":0.9978,\"pH\":3.51,\"sulphates\":0.56,\"alcohol\":9.4}]}"
```

> [!NOTE]
> Notice: the consumer code (step 5) **never references a run_id**. We can roll back to v1 with one `/promote` call without changing any service.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Visualize the Registry in the UI

Open [http://localhost:5000/#/models](http://localhost:5000/#/models). You should see:

```text
Models
└── WineQualityPredictor
    ├── Latest Version : 2 (Production)
    ├── v1   stage: Archived
    └── v2   stage: Production   ← currently served by /predict-production
```

Click on a version to see its source `run_id`, the metrics that were attached to that run, the signature, and direct links back to the Tracking experiment.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-13"></a>

## 13. Tear down

```bash
docker compose down
```

Persistent volumes `mlflow-db` and `mlflow-artifacts` keep the registered model across reboots — your Production tag survives a `docker compose down/up`. To wipe everything, also pass `-v`:

```bash
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-14"></a>

## 14. Recap

- The Registry lives next to Tracking on the same MLflow server (different UI tab: **Models**).
- `mlflow.register_model(uri, name)` — turns a run-artifact into a named, versioned model.
- `MlflowClient.transition_model_version_stage(name, version, stage, archive_existing_versions=True)` — governs the lifecycle.
- Consumers load with `models:/<name>/<stage>` — **stable URI**, decoupled from runs.
- Decoupling = safe rollbacks, A/B promotions, and zero-code-change deployments.

> [!IMPORTANT]
> Next chapter (18) introduces the **MLflow CLI** — the same operations, but from the terminal: list experiments, search runs, download artifacts, serve a registered model from the command line.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 17 — Model Registry &amp; <code>MlflowClient</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
