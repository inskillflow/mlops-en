<a id="top"></a>

# Chapter 14 — Today's topic: model **signature** + **input example** + `mlflow.sklearn.log_model()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Why do you need a signature?](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The lines we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Updated `streamlit/app/app.py`](#section-7) |
| 8 | [Run the stack](#section-8) |
| 9 | [Visualize signature & input example in the UI](#section-9) |
| 10 | [`infer_signature` vs manual `ModelSignature`](#section-10) |
| 11 | [Mini exercise — break the signature on purpose](#section-11) |
| 12 | [Tear down](#section-12) |
| 13 | [Recap](#section-13) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack and we **turn off model autolog** (Chapter 13) so we can take **manual control** of:

- the **signature** of the model — a typed contract: *"this model accepts these 11 columns of `double` and returns 1 column of `long`"*;
- the **input example** — a small payload (5 rows) saved alongside the model, that anyone can copy/paste to test the inference;
- the call to `mlflow.sklearn.log_model(model, "model", signature=..., input_example=...)`.

We also keep autolog **for params and metrics** (we just disable `log_models` and `log_model_signatures` so we don't get duplicates).

> [!IMPORTANT]
> A signed model with an input example is what makes a model **deployable**. Without a signature, MLflow can't validate input shapes at serving time; without an input example, downstream consumers have to read your code to understand what payload to send. Both should be considered **mandatory** in production.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Why do you need a signature?

A **signature** is a typed schema:

```text
inputs:
  - name: fixed acidity         type: double
  - name: volatile acidity      type: double
  - name: citric acid           type: double
  ... (11 columns)
outputs:
  - type: long
```

It is saved as a JSON file inside the model artifact (`model/MLmodel`). When you serve the model with `mlflow models serve`, MLflow:

1. Validates the JSON payload against the signature → bad input gets a clear `400 Bad Request`.
2. Renders the signature in the UI under the model artifact.
3. Generates Pydantic models / OpenAPI docs in some serving modes.

> [!NOTE]
> An **input example** is **not** a signature. It's just a serialized sample (a `JSON` file). The signature describes the **type contract**; the input example shows **a real payload** that respects that contract. Both go together.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.models.infer_signature(X, y_pred)` | **NEW today.** Auto-derive a `ModelSignature` from a numpy/pandas input and a prediction array. |
| `mlflow.models.signature.ModelSignature(inputs, outputs)` | **NEW today.** Manually define a signature when `infer_signature` is wrong (rare). |
| `mlflow.types.schema.Schema([ColSpec(...)])` | **NEW today.** Building blocks for `ModelSignature`. |
| `mlflow.sklearn.log_model(sk_model, artifact_path, signature=, input_example=)` | **NEW today.** Save the model under `artifacts/<artifact_path>/`. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The lines we add today

```python
from mlflow.models.signature import infer_signature

mlflow.sklearn.autolog(
    log_input_examples=False,    # NEW: turn off auto input example
    log_model_signatures=False,  # NEW: turn off auto signature
    log_models=False,            # NEW: turn off auto model save
)

# inside the run, after fit() and predict():
signature = infer_signature(test_x, preds)              # NEW
input_example = test_x.head(5)                          # NEW (5 rows DataFrame)

mlflow.sklearn.log_model(                               # NEW
    sk_model=lr,
    artifact_path="model",
    signature=signature,
    input_example=input_example,
)
```

That's it. Four new lines (plus the disable flags), and we now have a **deployable** model artifact.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap14-saving-mlflow-models-with-signature-and-input-example/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py            ← signature + log_model added
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py             ← shows the input_example back to the user
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Modified code — `fastapi/app/main.py`

```python
import io
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from mlflow.models.signature import infer_signature           # NEW
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap14_signature"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

mlflow.sklearn.autolog(             # autolog still on for params/metrics ...
    log_input_examples=False,       # NEW: but model+signature off
    log_model_signatures=False,     # NEW
    log_models=False,               # NEW
)

app = FastAPI(title="Chapter 14 — signature + input_example")


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
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        signature = infer_signature(test_x, preds)              # NEW
        input_example = test_x.head(5)                          # NEW

        mlflow.sklearn.log_model(                               # NEW
            sk_model=lr,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "test_metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "input_example": input_example.to_dict(orient="records"),
        "signature": str(signature),
    }
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Updated `streamlit/app/app.py`

```python
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("Chapter 14 — signature + input_example")

col1, col2 = st.columns(2)
alpha = col1.number_input("alpha", 0.0, 1.0, 0.5, 0.1)
l1_ratio = col2.number_input("l1_ratio", 0.0, 1.0, 0.5, 0.1)

if st.button("Train", type="primary"):
    r = requests.post(f"{API_URL}/train", json={"alpha": alpha, "l1_ratio": l1_ratio}, timeout=180)
    r.raise_for_status()
    data = r.json()
    st.success(f"Run `{data['run_name']}` created in `{data['experiment']}`")

    m = data["test_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test RMSE", f"{m['rmse']:.4f}")
    c2.metric("Test MAE", f"{m['mae']:.4f}")
    c3.metric("Test R²", f"{m['r2']:.4f}")

    with st.expander("Model signature attached to the run"):
        st.code(data["signature"], language="text")

    with st.expander("Input example (5 rows)"):
        st.dataframe(data["input_example"])
        st.markdown(
            "Copy any row → POST it to the model server "
            "`mlflow models serve -m runs:/<run_id>/model -p 1234`"
        )
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run the stack

```bash
cd chap14-saving-mlflow-models-with-signature-and-input-example
docker compose up --build
```

Trigger one training:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Visualize signature & input example in the UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap14_signature`** → click the run → tab **Artifacts** → folder **`model/`**.

You should now see:

```text
model/
├── MLmodel              ← contains the signature block
├── model.pkl
├── conda.yaml
├── python_env.yaml
├── requirements.txt
└── input_example.json   ← 5 sample rows
```

Click on `MLmodel` and scroll to the bottom — there is a `signature:` block that lists the 11 input columns with their types and the output type. Click `input_example.json` — it shows the 5-row sample as JSON.

In the **right panel** of the run view, MLflow renders a small **"Make Predictions"** code snippet using the input example. Copy/paste it directly into a Python notebook to call your model.

> [!NOTE]
> The signature is stored **inside `MLmodel`** (a YAML file). The input example is a **standalone JSON file**. They serve two different purposes (contract vs. sample) but they're meant to live together.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. `infer_signature` vs manual `ModelSignature`

99 % of the time, `infer_signature` is enough. The remaining 1 % is when:

- your **input** is a numpy array without column names → infer_signature uses indices instead of names;
- you want to **rename** columns (e.g. domain-friendly names) before exposing them via REST;
- you have **optional** inputs.

In those cases, build the signature by hand:

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
    ColSpec("double", "fixed_acidity"),
    ColSpec("double", "volatile_acidity"),
    ColSpec("double", "citric_acid"),
    # ... 11 columns total
])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

Pass that `signature` to `log_model(...)` exactly the same way.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Mini exercise — break the signature on purpose

After running a training, copy the `run_id` from the response.

In another terminal, install MLflow locally and serve the model:

```bash
pip install "mlflow==2.16.2" "scikit-learn==1.5.2"
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "runs:/<RUN_ID>/model" -p 1234 --no-conda
```

Send a **valid** payload (matching the signature):

```bash
curl -X POST http://localhost:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"fixed acidity":7.4,"volatile acidity":0.7,"citric acid":0.0,"residual sugar":1.9,"chlorides":0.076,"free sulfur dioxide":11.0,"total sulfur dioxide":34.0,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}]}'
```

Now drop one column (e.g. `alcohol`) and resend → you should get a `400 Bad Request` from MLflow saying the schema doesn't match. **That's the signature working.** Without it, the call would have crashed deep inside scikit-learn instead.

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

- A **signature** is the typed schema of a model's inputs/outputs. Saved in `MLmodel`.
- An **input example** is a sample payload (JSON). Saved in `input_example.json`.
- `infer_signature(X, y_pred)` derives the signature automatically; use `ModelSignature` only when you need full control.
- `mlflow.sklearn.log_model(sk_model, "model", signature=, input_example=)` writes the deployable artifact.
- Together, signature + input example make a model **servable** via `mlflow models serve`.
- The Docker stack is unchanged.

> [!IMPORTANT]
> Next chapter (15) keeps the same stack and shows how to **wrap a custom prediction logic** around a sklearn model using `mlflow.pyfunc.PythonModel`. Useful when you need preprocessing, post-processing, or to bundle two estimators in one logical model.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 14 — signature + input_example + <code>log_model</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
