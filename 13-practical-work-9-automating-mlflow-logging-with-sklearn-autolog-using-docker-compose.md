<a id="top"></a>

# Chapter 13 — Today's topic: `mlflow.sklearn.autolog()`

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [What is autolog?](#section-2) |
| 3 | [MLflow functions introduced](#section-3) |
| 4 | [The line we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Streamlit unchanged](#section-7) |
| 8 | [Run the stack](#section-8) |
| 9 | [Visualize what autolog captured](#section-9) |
| 10 | [Manual vs Autolog — side-by-side](#section-10) |
| 11 | [Mini exercise — disable parts of autolog](#section-11) |
| 12 | [Tear down](#section-12) |
| 13 | [Recap](#section-13) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack and we add **one new MLflow line**:

```python
mlflow.sklearn.autolog()
```

That single line, placed **before** `model.fit(...)`, automatically logs:

- all hyperparameters of the estimator (every constructor argument),
- the training metrics (`training_score`, `training_mse`, etc.),
- the trained **model itself** (with signature + input example, MLflow infers them),
- a **post-training section** (feature importance, etc., when applicable),
- and a few system tags.

After this chapter we **delete most of the manual `log_param` / `log_metric` / `log_model` calls** and let MLflow do it automatically.

> [!IMPORTANT]
> Autolog is the easiest way to log a scikit-learn run, but it's also a **black box**. You should know what it captures, what it doesn't, and how to disable parts of it (we'll see this in section 10).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. What is autolog?

`mlflow.sklearn.autolog()` patches scikit-learn at runtime so that every call to `estimator.fit(...)` opens or attaches to an MLflow run and logs:

| Category | What gets logged |
|---|---|
| **Params** | All constructor args of the estimator (`alpha`, `l1_ratio`, `random_state`, `max_iter`…) |
| **Metrics** | `training_score`, `training_mse`, `training_mae`, `training_r2_score` (and friends, depending on estimator) |
| **Model** | A `MLmodel` artifact under `model/` with the pickled estimator |
| **Signature** | Inferred automatically from the input passed to `fit` |
| **Input example** | First 5 rows of the training data |
| **System tags** | `estimator_class`, `estimator_name`, `mlflow.runName`, `mlflow.source.type=LOCAL` |

There is **one** flavor per ML library: `mlflow.sklearn.autolog`, `mlflow.tensorflow.autolog`, `mlflow.pytorch.autolog`, `mlflow.xgboost.autolog`… or the generic `mlflow.autolog()` that turns them all on at once.

> [!NOTE]
> Autolog only logs the **training** score (computed on `train_x`). It does **not** know about your test set. Test metrics (RMSE on `test_y`) still have to be logged manually with `mlflow.log_metric(...)`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow functions introduced

| Function | Description |
|---|---|
| `mlflow.sklearn.autolog(...)` | **NEW today.** Patch scikit-learn to auto-log everything around `fit`. |
| `mlflow.autolog()` | (Bonus) Generic autolog — turns it on for **all** supported libraries detected in your imports. |

Useful arguments of `mlflow.sklearn.autolog`:

| Argument | Default | Use |
|---|---|---|
| `log_input_examples` | `False` | Log first 5 rows of training data |
| `log_model_signatures` | `True` | Infer & attach a signature |
| `log_models` | `True` | Save the trained estimator as an artifact |
| `disable` | `False` | Set `True` to opt out for a single block |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The line we add today

```python
mlflow.sklearn.autolog(log_input_examples=True)   # NEW
```

Place this call **before** `model.fit(...)`. It can be called once at module load (top of `main.py`) — it only patches sklearn, doesn't open a run.

We **also delete** the manual calls to `log_param("alpha")`, `log_param("l1_ratio")`, the train-side metric calls, and the `mlflow.sklearn.log_model(...)` (autolog handles all of these). What remains manual:

- `log_metric("rmse", rmse)` on the **test** set,
- `log_metric("mae", mae)` on the **test** set,
- `log_metric("r2", r2)` on the **test** set,
- the few tags we still want.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap13-automating-mlflow-logging-with-sklearn-autolog/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py            ← much shorter than chap11/12
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py             ← unchanged from chap11
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
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap13_autolog"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

mlflow.sklearn.autolog(           # NEW: patch sklearn at import time
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
)

app = FastAPI(title="Chapter 13 — autolog")


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
        mlflow.set_tag("engineering", "ML platform")
        mlflow.set_tag("release.version", "2.0")

        # No log_param() here — autolog will capture alpha, l1_ratio, etc.
        # No log_model() here — autolog will save the estimator under model/.
        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Test-set metrics still logged by hand (autolog only logs training metrics)
        preds = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "test_metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "note": "alpha, l1_ratio, training_score and the model itself were autologged.",
    }
```

> [!IMPORTANT]
> Notice the file is **shorter** than `chap10/11/12`. Autolog removes ~10 lines of boilerplate. The 3 explicit `log_metric("test_*")` calls remain because **autolog never sees the test set** — only `fit(train_x, train_y)`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Streamlit unchanged

The Streamlit app stays the same as in chap11 (one form: alpha, l1_ratio, run_name → train button). The only visible change is in the response payload (`test_metrics` instead of `metrics`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run the stack

```bash
cd chap13-automating-mlflow-logging-with-sklearn-autolog
docker compose up --build
```

Trigger 2-3 trainings:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.3,\"l1_ratio\":0.3}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Visualize what autolog captured

Open [http://localhost:5000](http://localhost:5000) → experiment **`wine_quality_chap13_autolog`** → pick a run.

Without writing a single explicit `log_param`, you should see in the **Parameters** section:

```text
alpha            0.3
copy_X           True
fit_intercept    True
l1_ratio         0.3
max_iter         1000
positive         False
precompute       False
random_state     42
selection        cyclic
tol              0.0001
warm_start       False
```

In **Metrics**:

```text
training_mae        ...
training_mse        ...
training_r2_score   ...
training_rmse       ...
test_rmse           ...   ← we logged this manually
test_mae            ...   ← we logged this manually
test_r2             ...   ← we logged this manually
```

In **Artifacts**:

```text
model/
├── MLmodel              ← MLflow model spec (flavor: sklearn)
├── model.pkl            ← the pickled ElasticNet estimator
├── conda.yaml           ← Conda env to deploy this model
├── python_env.yaml      ← pip alternative
├── requirements.txt
└── input_example.json   ← first 5 rows of train_x (we asked for it)
```

> [!NOTE]
> The **whole** `model/` folder was created automatically. We could now serve this model with `mlflow models serve -m runs:/<run_id>/model -p 1234` — we'll do exactly that in Chapter 16.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Manual vs Autolog — side-by-side

| Concern | Manual (chap09–11) | Autolog (chap13) |
|---|---|---|
| Lines of code | ~12 lines for params + metrics + model | **1 line** at top of file |
| Hyperparameters | Whatever you remember to log | All constructor args (full transparency) |
| Training metrics | Whatever you compute & log | Auto-computed on training set |
| **Test metrics** | Manual | **Manual (autolog never sees the test set)** |
| Model artifact | Manual `mlflow.sklearn.log_model(lr)` | Auto-saved under `model/` |
| Signature & input example | Manual `infer_signature` | Auto-inferred |
| Tags | Manual `set_tag` | Manual (a few system tags only) |
| Multi-step pipelines (`Pipeline`, `GridSearchCV`) | You log everything yourself | Autolog logs **the parent fit** + best estimator |
| Risk | You forget something | The autolog patch may collide with custom callbacks |

**Rule of thumb**: use **autolog for the 80 % case** (single estimator, simple training loop) and fall back to manual logging when you need precise control (custom training loops, federated training, complex pipelines).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Mini exercise — disable parts of autolog

Edit `main.py`:

```python
mlflow.sklearn.autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,        # <-- model NOT saved automatically
)
```

Restart `fastapi`, run a training, and check the run in the UI:
- Parameters and metrics: still there.
- **Artifacts**: empty. No `model/` folder.

Now you know how to opt out and re-introduce manual `mlflow.sklearn.log_model(...)` if you need a custom artifact path.

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

- `mlflow.sklearn.autolog()` is **one line** that replaces ~10 lines of manual logging.
- It captures **params, training metrics, the model, signature, input example, system tags**.
- Test-set metrics still need explicit `log_metric` calls (autolog only sees `fit`).
- It works on `Pipeline` and `GridSearchCV` too — the parent `fit` is logged.
- The Docker stack is unchanged.

> [!IMPORTANT]
> Next chapter (14) keeps the same stack and shows how to **manually** define a model **signature** and an **input example** with `infer_signature` and `ModelSignature`, then attach them to a custom `mlflow.sklearn.log_model(...)` call. Useful when autolog is off or when the auto-inferred signature is wrong.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 13 — <code>mlflow.sklearn.autolog()</code></strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
