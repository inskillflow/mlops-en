<a id="top"></a>

# Chapter 12 — Today's topic: launching **multiple runs** and **multiple experiments** in one shot

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Why loops?](#section-2) |
| 3 | [MLflow patterns introduced](#section-3) |
| 4 | [The lines we add today](#section-4) |
| 5 | [Project structure](#section-5) |
| 6 | [Modified code — `fastapi/app/main.py`](#section-6) |
| 7 | [Updated `streamlit/app/app.py` (grid + sweep buttons)](#section-7) |
| 8 | [Run the stack](#section-8) |
| 9 | [Visualize multiple runs / experiments](#section-9) |
| 10 | [Mini exercise — best of 3 families](#section-10) |
| 11 | [`mlflow.start_run(nested=True)` — bonus](#section-11) |
| 12 | [Tear down](#section-12) |
| 13 | [Recap](#section-13) |

---

<a id="section-1"></a>

## 1. Objective

Today we keep the same Docker stack and we expose **two new endpoints** in FastAPI:

- `POST /train-grid` — runs **multiple runs** in a **single experiment** (a grid of `(alpha, l1_ratio)` values for ElasticNet);
- `POST /train-sweep` — runs **multiple runs across multiple experiments** (one experiment per model family: ElasticNet, Ridge, Lasso).

In both cases, every iteration of the loop opens its own `mlflow.start_run(...)` context manager, logs its params/metrics/tags, and closes cleanly.

> [!IMPORTANT]
> Up to Chapter 11, every API call → 1 run. From this chapter on, **one API call → N runs**, possibly across **M experiments**. This is exactly the shape of a **grid search** or **hyperparameter sweep**, recorded entirely in MLflow.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Why loops?

Picking the best `alpha` by hand is tedious. The natural workflow is:

1. Define a list of candidate hyperparameters (e.g. `alpha ∈ {0.1, 0.5, 0.9}`, `l1_ratio ∈ {0.1, 0.5, 0.9}` → 9 combinations).
2. Loop over them; for each, train + log to MLflow.
3. Open the MLflow UI, sort by `metrics.rmse` ascending, take the top one.

Same idea for **multiple algorithms**: ElasticNet, Ridge, Lasso. Each algorithm gets its **own MLflow experiment** so the runs are kept logically separate, but you still launch the whole sweep with one HTTP call.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. MLflow patterns introduced

| Pattern | Description |
|---|---|
| `for ... : with mlflow.start_run(...) as run:` | **NEW today.** Loop pattern — N runs in 1 script. |
| `mlflow.set_experiment("exp_X"); for ... : start_run(...)` | **NEW today.** Per-iteration experiment switch — M experiments in 1 script. |
| `mlflow.start_run(nested=True)` | (Bonus, end of chapter) Open a child run **inside** an active parent run. |

> [!NOTE]
> The trick that makes the loop pattern clean is the **context manager**: every iteration uses `with mlflow.start_run(...):`. When the `with` block ends (normally or because of an exception), MLflow automatically closes the run. So even if iteration #4 crashes, runs #1–#3 are saved and #4 is marked `FAILED`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The lines we add today

The "grid" loop:

```python
for alpha, l1_ratio in product(ALPHAS, L1_RATIOS):       # NEW
    with mlflow.start_run(run_name=f"a{alpha}_l{l1_ratio}"):
        # ... train + log_param + log_metric ...
```

The "sweep across experiments" loop:

```python
for family, model_factory in MODEL_FAMILIES.items():       # NEW
    mlflow.set_experiment(f"wine_quality_chap12_{family}")  # NEW (per family)
    for alpha in ALPHAS:                                    # NEW
        with mlflow.start_run(run_name=f"{family}_a{alpha}"):
            # ... train + log_param + log_metric ...
```

That's it. **Loops + context manager.** The rest is just data.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Project structure

```text
chap12-running-multiple-mlflow-runs-and-experiments/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py            ← /train-grid + /train-sweep added
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py             ← 2 new buttons
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Modified code — `fastapi/app/main.py`

```python
import io
import os
from itertools import product

import mlflow
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_GRID = "wine_quality_chap12_grid"
ALPHAS = [0.1, 0.5, 0.9]
L1_RATIOS = [0.1, 0.5, 0.9]

MODEL_FAMILIES = {                              # NEW
    "elasticnet": lambda a: ElasticNet(alpha=a, l1_ratio=0.5, random_state=42),
    "ridge": lambda a: Ridge(alpha=a, random_state=42),
    "lasso": lambda a: Lasso(alpha=a, random_state=42),
}

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 12 — multiple runs & multiple experiments")


def load_data() -> pd.DataFrame:
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), sep=";")


def split(data: pd.DataFrame):
    train_df, test_df = train_test_split(data, random_state=40)
    return (
        train_df.drop(["quality"], axis=1),
        test_df.drop(["quality"], axis=1),
        train_df["quality"],
        test_df["quality"],
    )


def eval_metrics(actual, pred):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2


@app.post("/train-grid")
def train_grid():
    """Run a 3x3 ElasticNet grid in a single experiment."""
    train_x, test_x, train_y, test_y = split(load_data())
    mlflow.set_experiment(EXPERIMENT_GRID)

    runs = []
    for alpha, l1_ratio in product(ALPHAS, L1_RATIOS):                    # NEW
        with mlflow.start_run(run_name=f"a{alpha}_l{l1_ratio}"):           # NEW
            mlflow.set_tag("sweep", "elasticnet-grid")
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)
            preds = lr.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, preds)
            mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            run_info = mlflow.active_run().info
            runs.append({
                "run_id": run_info.run_id,
                "run_name": run_info.run_name,
                "alpha": alpha, "l1_ratio": l1_ratio,
                "rmse": rmse, "mae": mae, "r2": r2,
            })

    runs.sort(key=lambda r: r["rmse"])
    return {
        "experiment": EXPERIMENT_GRID,
        "n_runs": len(runs),
        "best": runs[0],
        "all": runs,
    }


@app.post("/train-sweep")
def train_sweep():
    """Run 3 alphas across 3 model families = 3 experiments x 3 runs."""
    train_x, test_x, train_y, test_y = split(load_data())

    summary = {}
    for family, factory in MODEL_FAMILIES.items():                         # NEW
        exp_name = f"wine_quality_chap12_{family}"
        mlflow.set_experiment(exp_name)                                    # NEW
        family_runs = []
        for alpha in ALPHAS:                                               # NEW
            with mlflow.start_run(run_name=f"{family}_a{alpha}"):           # NEW
                mlflow.set_tag("sweep", "multi-family")
                mlflow.set_tag("model.family", family)
                model = factory(alpha)
                model.fit(train_x, train_y)
                preds = model.predict(test_x)
                rmse, mae, r2 = eval_metrics(test_y, preds)
                mlflow.log_param("alpha", alpha)
                mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
                run_info = mlflow.active_run().info
                family_runs.append({
                    "run_id": run_info.run_id,
                    "run_name": run_info.run_name,
                    "alpha": alpha,
                    "rmse": rmse, "mae": mae, "r2": r2,
                })
        family_runs.sort(key=lambda r: r["rmse"])
        summary[family] = {"experiment": exp_name, "best": family_runs[0], "all": family_runs}

    overall_best = min(
        (s["best"] | {"family": fam} for fam, s in summary.items()),
        key=lambda r: r["rmse"],
    )
    return {
        "n_experiments": len(summary),
        "n_runs": sum(len(s["all"]) for s in summary.values()),
        "overall_best": overall_best,
        "by_family": summary,
    }
```

> [!IMPORTANT]
> Notice that **each iteration calls `mlflow.set_experiment(...)` only when the family changes** in `/train-sweep`. Calling `set_experiment` is idempotent (no-op if already on that experiment) but it's cleaner to keep it at the boundary.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Updated `streamlit/app/app.py` (grid + sweep buttons)

```python
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("Chapter 12 — multiple runs & multiple experiments")

st.markdown(
    "Two batch endpoints:\n"
    "- **Grid** runs 9 ElasticNet configs in one experiment.\n"
    "- **Sweep** runs 3 alphas × 3 model families = **3 experiments × 3 runs**."
)

c1, c2 = st.columns(2)

if c1.button("Run grid (9 runs)", type="primary"):
    with st.spinner("Training 9 ElasticNet runs..."):
        r = requests.post(f"{API_URL}/train-grid", timeout=600)
        r.raise_for_status()
        data = r.json()
    st.success(f"{data['n_runs']} runs logged in `{data['experiment']}`")
    st.markdown(f"**Best run** (lowest RMSE): `{data['best']['run_name']}`")
    st.dataframe(data["all"])

if c2.button("Run sweep (9 runs / 3 experiments)"):
    with st.spinner("Training 3 model families..."):
        r = requests.post(f"{API_URL}/train-sweep", timeout=600)
        r.raise_for_status()
        data = r.json()
    st.success(f"{data['n_runs']} runs across {data['n_experiments']} experiments")
    best = data["overall_best"]
    st.markdown(
        f"**Overall best**: family **{best['family']}** with alpha={best['alpha']} → "
        f"RMSE={best['rmse']:.4f}"
    )
    for family, summary in data["by_family"].items():
        with st.expander(f"{family} — best alpha={summary['best']['alpha']}"):
            st.dataframe(summary["all"])
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Run the stack

```bash
cd chap12-running-multiple-mlflow-runs-and-experiments
docker compose up --build
```

Trigger both batches:

```bash
curl -X POST http://localhost:8000/train-grid   # 9 runs in 1 experiment
curl -X POST http://localhost:8000/train-sweep  # 9 runs across 3 experiments
```

Or use the two buttons in Streamlit at [http://localhost:8501](http://localhost:8501).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Visualize multiple runs / experiments

Open [http://localhost:5000](http://localhost:5000). On the left sidebar you should now see **four** experiments:

- `wine_quality_chap12_grid` (9 runs)
- `wine_quality_chap12_elasticnet` (3 runs)
- `wine_quality_chap12_ridge` (3 runs)
- `wine_quality_chap12_lasso` (3 runs)

In the **grid** experiment, sort by `metrics.rmse` ascending — the best `(alpha, l1_ratio)` combination is at the top. Select the 9 runs → **Compare** → MLflow draws a **parallel coordinates** chart that visually shows which `(alpha, l1_ratio)` band gives the best RMSE.

> [!IMPORTANT]
> The MLflow **search bar** lets you compare across experiments too: tick the experiments you want, then sort. This is exactly how teams pick a winner across model families.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Mini exercise — best of 3 families

1. Click **Run sweep** in Streamlit (or hit `/train-sweep`).
2. Read the response: it tells you the overall winner.
3. Verify in MLflow UI: open the winning experiment, locate the run with the lowest `rmse`. Should match.
4. Add `Lasso(alpha=2.0)` to the grid (open `MODEL_FAMILIES` and tweak), restart `fastapi`, run again, and check that the new alpha appears in the `lasso` experiment as a 4th run.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. `mlflow.start_run(nested=True)` — bonus

Sometimes you want **one parent run that summarizes the whole grid** plus **N child runs** (one per cell of the grid). MLflow supports this natively:

```python
with mlflow.start_run(run_name="grid_parent") as parent:
    for alpha, l1_ratio in product(ALPHAS, L1_RATIOS):
        with mlflow.start_run(run_name=f"a{alpha}_l{l1_ratio}", nested=True) as child:
            # ... train, log params/metrics on child ...
            pass
    # at the end, log overall stats on parent
    mlflow.log_metric("best_rmse", min_rmse_seen)
```

In the MLflow UI, child runs are then **collapsed** under the parent (small triangle to expand). This is the cleanest way to record a hyperparameter sweep as **one logical experiment** with N sub-runs.

> [!NOTE]
> We will reuse `nested=True` later when we wrap a grid search inside the same run that registers the **winning model** to the Model Registry (Chapter 25).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. Tear down

```bash
docker compose down
```

> [!NOTE]
> The MLflow UI now contains **6 cumulative experiments** (chap07 → chap12) for a total of ~25 runs. We are now at the size where MLflow's **search and compare** features stop being optional.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-13"></a>

## 13. Recap

- Use a **`for` loop with a `with mlflow.start_run(...)` inside** to log many runs in one script.
- Switch experiments inside the loop with `mlflow.set_experiment(name)` to organize runs by **family** (per algorithm, per dataset version, etc.).
- `mlflow.start_run(nested=True)` opens a **child run** inside a parent run — perfect for hyperparameter sweeps.
- Sorting + parallel-coordinates in the UI become indispensable once you cross ~20 runs.
- The Docker stack is unchanged.

> [!IMPORTANT]
> Next chapter (13) keeps the same stack and adds the next single line: `mlflow.sklearn.autolog()`. We'll see how MLflow can log params/metrics/model **automatically**, removing 90 % of the boilerplate we wrote in Chapters 09–11.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 12 — multiple runs & multiple experiments</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
