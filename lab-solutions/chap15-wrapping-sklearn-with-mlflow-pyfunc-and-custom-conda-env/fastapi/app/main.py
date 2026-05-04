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

from app.wrapper import WineQualityWrapper           # NEW

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap15_pyfunc"
ARTIFACT_PATH = "wine_quality_pyfunc"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 15 - pyfunc wrapper + custom Conda env")


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


@app.get("/")
def root():
    return {"chapter": 15, "topic": "mlflow.pyfunc.PythonModel + custom Conda env"}


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
            joblib.dump(lr, sk_path)                                          # NEW

            artifacts = {"sklearn_model": sk_path}                            # NEW

            mlflow.pyfunc.log_model(                                          # NEW
                artifact_path=ARTIFACT_PATH,
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
        "artifact_path": ARTIFACT_PATH,
        "load_with": (
            f"mlflow.pyfunc.load_model("
            f"'runs:/{last.info.run_id}/{ARTIFACT_PATH}')"
        ),
    }
