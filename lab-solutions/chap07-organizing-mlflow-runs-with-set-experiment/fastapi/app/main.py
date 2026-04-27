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


@app.get("/health")
def health():
    return {"status": "ok"}


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
