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


@app.get("/")
def root():
    return {"chapter": "09", "topic": "log_param / log_metric", "mlflow": MLFLOW_URI}


@app.get("/health")
def health():
    return {"status": "ok"}


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
