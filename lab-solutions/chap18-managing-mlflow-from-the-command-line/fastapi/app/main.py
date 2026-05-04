"""Same Registry endpoints as Chapter 17 - we just need data to play with from the CLI."""
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

EXPERIMENT_NAME = "wine_quality_chap18_cli"
ARTIFACT_PATH = "wine_quality_pyfunc"
REGISTERED_MODEL_NAME = "WineQualityPredictor"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 18 - data producer for CLI demos")
client = MlflowClient()


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
    stage: str = "Production"
    archive_existing: bool = True


@app.get("/")
def root():
    return {"chapter": 18, "topic": "MLflow CLI"}


@app.get("/health")
def health():
    return {"status": "ok"}


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
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    return {
        "run_id": run_id,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "registered": {
            "name": mv.name,
            "version": int(mv.version),
            "current_stage": mv.current_stage,
        },
    }


@app.post("/promote")
def promote(req: PromoteRequest):
    if req.stage not in {"None", "Staging", "Production", "Archived"}:
        raise HTTPException(400, f"Invalid stage: {req.stage}")
    mv = client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=req.version,
        stage=req.stage,
        archive_existing_versions=req.archive_existing,
    )
    return {"name": mv.name, "version": int(mv.version), "new_stage": mv.current_stage}
