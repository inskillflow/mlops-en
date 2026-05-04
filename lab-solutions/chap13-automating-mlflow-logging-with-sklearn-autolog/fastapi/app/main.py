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

mlflow.sklearn.autolog(           # NEW
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
)

app = FastAPI(title="Chapter 13 - autolog")


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
    return {"chapter": 13, "topic": "mlflow.sklearn.autolog()"}


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
        mlflow.set_tag("engineering", "ML platform")
        mlflow.set_tag("release.version", "2.0")

        lr = ElasticNet(alpha=req.alpha, l1_ratio=req.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

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
