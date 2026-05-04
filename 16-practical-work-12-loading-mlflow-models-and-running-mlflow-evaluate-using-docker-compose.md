<a id="top"></a>

# Chapter 16 — Today's topic: `mlflow.pyfunc.load_model()` + `mlflow.evaluate()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [`load_model` in 30 seconds](#section-2) |
| 3 | [`mlflow.evaluate` in 30 seconds](#section-3) |
| 4 | [MLflow functions introduced](#section-4) |
| 5 | [The lines we add today](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [Modified code — `fastapi/app/main.py`](#section-7) |
| 8 | [Updated `streamlit/app/app.py`](#section-8) |
| 9 | [Run the stack](#section-9) |
| 10 | [Visualize the evaluation report in the UI](#section-10) |
| 11 | [Mini exercise — predict on a custom payload](#section-11) |
| 12 | [Tear down](#section-12) |
| 13 | [Recap](#section-13) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack and we add **two new MLflow capabilities**:

- **`mlflow.pyfunc.load_model("runs:/<run_id>/<artifact_path>")`** — load **back** the pyfunc model we logged in Chapter 15 and call `.predict(...)` on a fresh request;
- **`mlflow.evaluate(model_uri, data, targets="quality", model_type="regressor")`** — automatically compute a **full evaluation report** (RMSE, MAE, R², residuals plot, error distribution…) and log it to the run.

Two new FastAPI endpoints:

- `POST /train-and-evaluate` — trains a model **and** runs `mlflow.evaluate` on the test set;
- `POST /predict` — loads a model from a `run_id` you provide and returns predictions on a JSON payload.

> [!IMPORTANT]
> Up to Chapter 15, we always *trained and logged*. Today we close the loop: **train → log → load → predict → evaluate**. This is the full life-cycle of an MLflow model.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. `load_model` in 30 seconds

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("runs:/abc123.../wine_quality_pyfunc")
preds = model.predict(some_dataframe)   # numpy array or list
```

The URI accepts several forms:

| URI | Meaning |
|---|---|
| `runs:/<run_id>/<artifact_path>` | A specific run's artifact |
| `models:/<name>/<version>` | A version in the Model Registry (Chapter 17) |
| `models:/<name>/Production` | The current "Production" version |
| `s3://bucket/path/` or `./local/path/` | A raw path |

The model is loaded **once**, kept in memory, and `predict(...)` can be called as many times as needed. That's exactly how you'd build an inference service.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. `mlflow.evaluate` in 30 seconds

```python
result = mlflow.evaluate(
    model="runs:/<run_id>/wine_quality_pyfunc",
    data=test_df,           # DataFrame with the targets included
    targets="quality",      # column name
    model_type="regressor", # "classifier" / "regressor" / "question-answering" ...
    evaluators=["default"],
)
print(result.metrics)
```

It runs the model on `data`, compares predictions to the column `targets`, and **logs to the active run**:

- many metrics (`mean_squared_error`, `mean_absolute_error`, `r2_score`, `max_error`, `mean_on_target`…);
- artifacts: a residuals plot, an error distribution plot, a confusion matrix (for classifiers);
- the evaluation **dataset** itself (with predictions added);
- `result.metrics` returned in Python so you can use it in code.

> [!NOTE]
> `mlflow.evaluate` requires the model to be loadable via `mlflow.pyfunc.load_model(...)`. Our Chapter 15 wrapper qualifies. So does any model logged with `mlflow.sklearn.log_model(...)`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.pyfunc.load_model(uri)` | **NEW today.** Load a logged pyfunc model from any URI. |
| `mlflow.evaluate(model, data, targets, model_type, evaluators)` | **NEW today.** Auto-compute & log a full evaluation report. |
| `mlflow.get_artifact_uri(artifact_path)` | (Bonus) Resolve the URI of an artifact path inside the active run. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. The lines we add today

Inside the run, **right after `mlflow.pyfunc.log_model(...)`** of Chapter 15:

```python
artifact_uri = mlflow.get_artifact_uri("wine_quality_pyfunc")          # NEW

eval_result = mlflow.evaluate(                                         # NEW
    model=artifact_uri,
    data=test_df,
    targets="quality",
    model_type="regressor",
    evaluators=["default"],
)
```

And in a **new `/predict` endpoint** (no run is active there — we just load and predict):

```python
loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/wine_quality_pyfunc")  # NEW
out = loaded.predict(input_df)                                            # NEW
```

That's it. **Three new lines**, plus the wiring.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap16-loading-mlflow-models-and-running-mlflow-evaluate/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py            ← /train-and-evaluate + /predict
│       └── wrapper.py         ← unchanged from chap15
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py             ← 2 tabs: train+evaluate / predict
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Modified code — `fastapi/app/main.py`

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
from mlflow.models.signature import infer_signature
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.wrapper import WineQualityWrapper

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap16_load_evaluate"
ARTIFACT_PATH = "wine_quality_pyfunc"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 16 — load_model + mlflow.evaluate")


def load_data() -> pd.DataFrame:
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep=";")


def eval_metrics(actual, pred):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2


def build_conda_env() -> dict:
    return {
        "channels": ["defaults"],
        "dependencies": [
            "python=3.12",
            "pip",
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


class PredictRequest(BaseModel):
    run_id: str
    rows: list[dict]   # list of column-name -> value dicts


@app.post("/train-and-evaluate")
def train_and_evaluate(req: TrainRequest):
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
        mlflow.log_params({"alpha": req.alpha, "l1_ratio": req.l1_ratio})
        mlflow.log_metrics({"manual_rmse": rmse, "manual_mae": mae, "manual_r2": r2})

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

        artifact_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)             # NEW
        eval_result = mlflow.evaluate(                                     # NEW
            model=artifact_uri,
            data=test_df,
            targets="quality",
            model_type="regressor",
            evaluators=["default"],
        )

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "manual_metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "evaluate_metrics": {k: float(v) for k, v in eval_result.metrics.items()},
        "artifact_uri": artifact_uri,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Load a previously-trained pyfunc model and predict on the given rows."""
    try:
        model_uri = f"runs:/{req.run_id}/{ARTIFACT_PATH}"
        loaded = mlflow.pyfunc.load_model(model_uri)                       # NEW
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not loadable: {e}")

    df = pd.DataFrame(req.rows)
    preds = loaded.predict(df)
    return {
        "model_uri": model_uri,
        "n_rows": len(df),
        "predictions": list(preds),
    }
```

> [!IMPORTANT]
> Notice we keep both **manual** metrics (`manual_rmse`, `manual_mae`, `manual_r2`) and **`mlflow.evaluate`** metrics in the same run. The latter are richer (include `max_error`, residuals plot, etc.) but we want to confirm they match the manual ones. That's a good defensive habit.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Updated `streamlit/app/app.py`

```python
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("Chapter 16 — load_model + mlflow.evaluate")

tab_train, tab_predict = st.tabs(["Train + evaluate", "Predict from run"])

with tab_train:
    col1, col2 = st.columns(2)
    alpha = col1.number_input("alpha", 0.0, 1.0, 0.4, 0.1, key="t_a")
    l1_ratio = col2.number_input("l1_ratio", 0.0, 1.0, 0.4, 0.1, key="t_l")
    if st.button("Train + evaluate", type="primary"):
        r = requests.post(
            f"{API_URL}/train-and-evaluate",
            json={"alpha": alpha, "l1_ratio": l1_ratio},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        st.success(f"Run `{data['run_name']}` ({data['run_id']})")
        st.markdown("**Manual metrics:**")
        st.json(data["manual_metrics"])
        st.markdown("**`mlflow.evaluate` metrics:**")
        st.json(data["evaluate_metrics"])
        st.code(f"Save this run_id to predict later:\n{data['run_id']}")

with tab_predict:
    run_id = st.text_input("run_id (paste from the train tab)")
    sample = {
        "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
        "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
        "sulphates": 0.56, "alcohol": 9.4,
    }
    payload_text = st.text_area(
        "JSON payload (a list of rows)",
        value=__import__("json").dumps([sample], indent=2),
        height=260,
    )
    if st.button("Predict"):
        rows = __import__("json").loads(payload_text)
        r = requests.post(
            f"{API_URL}/predict",
            json={"run_id": run_id.strip(), "rows": rows},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        st.success(f"{data['n_rows']} predictions")
        st.json(data)
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Run the stack

```bash
cd chap16-loading-mlflow-models-and-running-mlflow-evaluate
docker compose up --build
```

Train + evaluate:

```bash
curl -X POST http://localhost:8000/train-and-evaluate \
  -H "Content-Type: application/json" \
  -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"
```

Note the `run_id` in the response. Now predict on a custom row:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"<RUN_ID>\",\"rows\":[{\"fixed acidity\":7.4,\"volatile acidity\":0.7,\"citric acid\":0.0,\"residual sugar\":1.9,\"chlorides\":0.076,\"free sulfur dioxide\":11.0,\"total sulfur dioxide\":34.0,\"density\":0.9978,\"pH\":3.51,\"sulphates\":0.56,\"alcohol\":9.4}]}"
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Visualize the evaluation report in the UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap16_load_evaluate`** → click the run.

In **Metrics** you should now see (besides the manual ones):

```text
mean_squared_error              ...
root_mean_squared_error         ...
mean_absolute_error             ...
r2_score                        ...
max_error                       ...
mean_on_target                  ...
sum_on_target                   ...
```

In **Artifacts**, a new folder appears:

```text
eval_results_table.json         ← raw evaluation result
shap/                           ← (when shap is installed) feature importances
plots/
├── residuals.png
└── prediction_distribution.png
```

> [!NOTE]
> `mlflow.evaluate` is the **same machinery** the MLflow UI uses behind the scenes when you click "Evaluate" on a model. Calling it from code lets you snapshot the report in the same run as the training, perfectly time-aligned.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Mini exercise — predict on a custom payload

1. Train via the Streamlit "Train + evaluate" tab.
2. Copy the `run_id` from the response.
3. Switch to the "Predict from run" tab, paste the `run_id`, edit the JSON payload (e.g. set `alcohol: 14.0`), click **Predict**.
4. The model returns a quality score in [3, 9] (rounded to 1 decimal — remember our Chapter 15 wrapper applies these business rules).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Tear down

```bash
docker compose down
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-13"></a>

## 13. Recap

- `mlflow.pyfunc.load_model("runs:/<run_id>/<artifact_path>")` reverses the `log_model` of Chapter 15.
- The model URI scheme also supports the **Model Registry** (`models:/<name>/<version>` or `models:/<name>/Production`).
- `mlflow.evaluate(model, data, targets, model_type)` produces a full evaluation report in **one line**.
- Together they close the **train → log → load → predict → evaluate** loop.
- The Docker stack is unchanged.

> [!IMPORTANT]
> Next chapter (17) keeps the same stack and adds the **Model Registry**: register a model under a name, manage versions, transition between `None / Staging / Production / Archived`, and resolve the URI `models:/WineQualityPredictor/Production` from any consumer.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 16 — <code>load_model</code> + <code>mlflow.evaluate</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
