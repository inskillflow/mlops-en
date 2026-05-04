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

app = FastAPI(title="Chapter 12 - multiple runs & multiple experiments")


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


@app.get("/")
def root():
    return {"chapter": 12, "topic": "multiple runs & experiments"}


@app.get("/health")
def health():
    return {"status": "ok"}


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
        summary[family] = {
            "experiment": exp_name,
            "best": family_runs[0],
            "all": family_runs,
        }

    overall_best = min(
        ({**s["best"], "family": fam} for fam, s in summary.items()),
        key=lambda r: r["rmse"],
    )
    return {
        "n_experiments": len(summary),
        "n_runs": sum(len(s["all"]) for s in summary.values()),
        "overall_best": overall_best,
        "by_family": summary,
    }
