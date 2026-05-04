<a id="top"></a>

# Chapter 15 — Today's topic: `mlflow.pyfunc.PythonModel` wrapper + custom Conda env

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Why wrap a sklearn model with pyfunc?](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The wrapper class](#section-4) |
| 5 | [The custom Conda env](#section-5) |
| 6 | [Project structure](#section-6) |
| 7 | [Modified code — `fastapi/app/main.py`](#section-7) |
| 8 | [Streamlit unchanged](#section-8) |
| 9 | [Run the stack](#section-9) |
| 10 | [Visualize the pyfunc model in the UI](#section-10) |
| 11 | [Mini exercise — add a unit conversion in `predict`](#section-11) |
| 12 | [Tear down](#section-12) |
| 13 | [Recap](#section-13) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack and we **wrap our sklearn model in a custom Python class** that follows the `mlflow.pyfunc.PythonModel` contract. Then we save it with `mlflow.pyfunc.log_model(...)` and a **custom Conda environment**.

Why? Because the bare `mlflow.sklearn.log_model(...)` from Chapter 14 is fine when the input goes **directly** into `model.predict(X)`. As soon as you need:

- **preprocessing** (e.g. clip outliers, normalize),
- **post-processing** (e.g. round scores, convert units),
- to bundle **two models** in one logical artifact (e.g. a feature encoder + a regressor),

…you have to write a small Python class that defines `load_context` and `predict`. That class is the **pyfunc wrapper**.

> [!IMPORTANT]
> A pyfunc-wrapped model is what you ship to production. The wrapper is the **single source of truth** for "what does inference look like": it loads everything it needs (`load_context`) and exposes one Python function (`predict`). MLflow can serve it, batch-run it, register it… all transparently.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Why wrap a sklearn model with pyfunc?

Concretely, in our wine-quality example, suppose we want the prediction to be **rounded to one decimal** and **clipped to the valid range [3, 9]** before being returned to the API client. There are 2 places to do that:

1. In FastAPI, after calling `model.predict(...)`. ❌ The transformation lives outside the model artifact, so anyone serving the model elsewhere (e.g. with `mlflow models serve`) won't apply it.
2. **Inside a pyfunc wrapper.** ✅ The transformation **travels with the model**. Whoever loads the artifact gets the same behavior, no matter how they serve it.

Same idea for any **bundled preprocessing**: imputation, encoding, feature engineering. Bake it into the wrapper, log the wrapper, and the deployment is reproducible.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function / class | Description |
|---|---|
| `mlflow.pyfunc.PythonModel` | **NEW today.** Base class. Subclass it and override `load_context(self, context)` and `predict(self, context, model_input)`. |
| `mlflow.pyfunc.log_model(artifact_path, python_model=, artifacts=, conda_env=, signature=, input_example=)` | **NEW today.** Save the wrapper + its dependencies. |
| `joblib.dump(model, path)` | (Bonus) Serialize the underlying sklearn model to a `.pkl`. |
| `mlflow.pyfunc.load_model(uri)` | (Used in chap 16) Load back the wrapper to call `.predict()`. |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The wrapper class

The contract is two methods:

```python
import joblib
import mlflow.pyfunc
import numpy as np


class WineQualityWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):                   # called once at load time
        self.model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):           # called for every request
        raw = self.model.predict(model_input)          # numpy array
        clipped = np.clip(raw, 3.0, 9.0)               # business rule
        return np.round(clipped, 1).tolist()           # JSON-friendly
```

- `load_context` reads the file paths declared in the `artifacts={...}` dictionary. Here we point `sklearn_model` to a `joblib`-serialized ElasticNet.
- `predict` receives whatever the user passes (dict, DataFrame, numpy array). MLflow standardizes it. We do business logic and return a list/array.

> [!NOTE]
> The class is *just* a Python class. It must be **picklable** (no closures, no lambdas at module scope) so MLflow can serialize it. If you need helpers, define them at module level too.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. The custom Conda env

When MLflow serves the model, it spins up a **fresh Conda environment** containing only the dependencies you declared. We provide that env explicitly:

```python
import mlflow, sklearn, cloudpickle

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python=3.12",
        "pip",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                "joblib",
                "numpy",
                "pandas",
            ],
        },
    ],
    "name": "wine_quality_env",
}
```

> [!IMPORTANT]
> Versions matter. Pin the exact `mlflow`, `scikit-learn`, `cloudpickle` versions used at training time. Otherwise, the model that "worked yesterday" may fail to load tomorrow because of a sklearn ABI change.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Project structure

```text
chap15-wrapping-sklearn-with-mlflow-pyfunc-and-custom-conda-env/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py            ← uses the wrapper
│       └── wrapper.py         ← the pyfunc class lives here
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py             ← unchanged
```

> [!NOTE]
> We put the wrapper class in its **own module** (`wrapper.py`) so MLflow can serialize the class with `cloudpickle`. Keeping it in `main.py` would also work but pollutes the file.

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
from fastapi import FastAPI
from mlflow.models.signature import infer_signature
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.wrapper import WineQualityWrapper             # NEW

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap15_pyfunc"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 15 — pyfunc wrapper + custom Conda env")


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
            {
                "pip": [
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"cloudpickle=={cloudpickle.__version__}",
                    "joblib",
                    "numpy",
                    "pandas",
                ],
            },
        ],
        "name": "wine_quality_env",
    }


class TrainRequest(BaseModel):
    alpha: float = 0.4
    l1_ratio: float = 0.4
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
        mlflow.set_tag("artifact.type", "pyfunc-wrapper")

        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        preds = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)
        mlflow.log_params({"alpha": req.alpha, "l1_ratio": req.l1_ratio})
        mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})

        signature = infer_signature(test_x, preds)
        input_example = test_x.head(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            sk_path = os.path.join(tmpdir, "sklearn_model.pkl")
            joblib.dump(lr, sk_path)                                        # NEW

            artifacts = {"sklearn_model": sk_path}                          # NEW

            mlflow.pyfunc.log_model(                                        # NEW
                artifact_path="wine_quality_pyfunc",
                python_model=WineQualityWrapper(),
                artifacts=artifacts,
                conda_env=build_conda_env(),
                signature=signature,
                input_example=input_example,
            )

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "test_metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "artifact_path": "wine_quality_pyfunc",
        "load_with": f"mlflow.pyfunc.load_model('runs:/{last.info.run_id}/wine_quality_pyfunc')",
    }
```

And the wrapper itself:

```python
# fastapi/app/wrapper.py
import joblib
import mlflow.pyfunc
import numpy as np


class WineQualityWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["sklearn_model"])

    def predict(self, context, model_input):
        raw = self.model.predict(model_input)
        clipped = np.clip(raw, 3.0, 9.0)
        return np.round(clipped, 1).tolist()
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Streamlit unchanged

The Streamlit UI is the same as Chapter 14 (alpha + l1_ratio + Train button). The response now contains a `load_with` field — a one-liner Python snippet to load the model anywhere:

```python
import mlflow.pyfunc
m = mlflow.pyfunc.load_model("runs:/<run_id>/wine_quality_pyfunc")
m.predict(payload)   # returns a list of clipped, rounded scores
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Run the stack

```bash
cd chap15-wrapping-sklearn-with-mlflow-pyfunc-and-custom-conda-env
docker compose up --build
```

Trigger one training:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Visualize the pyfunc model in the UI

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap15_pyfunc`** → run → tab **Artifacts** → folder **`wine_quality_pyfunc/`**.

```text
wine_quality_pyfunc/
├── MLmodel                    ← flavor: python_function
├── conda.yaml                 ← OUR custom Conda env
├── python_env.yaml
├── python_model.pkl           ← cloudpickle of WineQualityWrapper instance
├── artifacts/
│   └── sklearn_model.pkl      ← the joblib-serialized ElasticNet
├── input_example.json
└── requirements.txt
```

Open `MLmodel` and look for `flavors:` — there is **only one flavor** (`python_function`). Compare with Chapter 14 where the same artifact had `flavors: [sklearn, python_function]`. With the wrapper, MLflow no longer knows it's a sklearn model — and that's the point: the wrapper hides it.

> [!IMPORTANT]
> Custom `predict` logic = custom flavor. From the consumer's point of view, this is just a `python_function`: load it, call `.predict(X)`, get rounded predictions. They don't need to know there's an ElasticNet inside.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Mini exercise — add a unit conversion in `predict`

Edit `fastapi/app/wrapper.py` so `predict` returns the score on a **0–100 scale** instead of 3–9:

```python
def predict(self, context, model_input):
    raw = self.model.predict(model_input)
    clipped = np.clip(raw, 3.0, 9.0)
    rescaled = (clipped - 3.0) / 6.0 * 100.0
    return np.round(rescaled, 1).tolist()
```

Restart `fastapi` (`docker compose restart fastapi`), re-train, and verify in the run's artifact folder that the new wrapper has been re-pickled. Predictions should now be in [0, 100].

> [!NOTE]
> This is the **whole point** of pyfunc: bake business logic into the artifact. A consumer who loads `runs:/<run_id>/wine_quality_pyfunc` next year will automatically get predictions on the 0–100 scale without us having to ship them separate code.

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

- `mlflow.pyfunc.PythonModel` is a 2-method base class: `load_context` (open files) + `predict` (apply logic).
- `mlflow.pyfunc.log_model(...)` saves the wrapper instance, the artifacts it depends on, the **custom Conda env**, the signature and the input example — all in one folder.
- A pyfunc model is the **deployable unit**: pre/post-processing travels with the weights.
- `joblib.dump(model, "x.pkl")` is the typical way to serialize sklearn objects passed as artifacts.
- The Docker stack is unchanged.

> [!IMPORTANT]
> Next chapter (16) keeps the same stack and shows how to **load back** a pyfunc model with `mlflow.pyfunc.load_model("runs:/<run_id>/wine_quality_pyfunc")` and call `.predict()` on a fresh request — the inverse operation of today.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 15 — pyfunc wrapper + custom Conda env</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
