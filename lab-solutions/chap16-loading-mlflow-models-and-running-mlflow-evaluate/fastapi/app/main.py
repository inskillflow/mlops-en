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
from mlflow.models.signature import infer_signature
from pydantic import BaseModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.wrapper import WineQualityWrapper

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_quality_chap16_load_evaluate"
ARTIFACT_PATH = "wine_quality_pyfunc"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/wine-quality/winequality-red.csv"
)

app = FastAPI(title="Chapter 16 - load_model + mlflow.evaluate")


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


class PredictRequest(BaseModel):
    run_id: str
    rows: list[dict]


@app.get("/")
def root():
    return {"chapter": 16, "topic": "load_model + mlflow.evaluate"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train-and-evaluate")
def train_and_evaluate(req: TrainRequest):
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
        mlflow.log_params({"alpha": req.alpha, "l1_ratio": req.l1_ratio})
        mlflow.log_metrics({"manual_rmse": rmse, "manual_mae": mae, "manual_r2": r2})

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

        artifact_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)             # NEW
        eval_result = mlflow.evaluate(                                     # NEW
            model=artifact_uri,
            data=test_df,
            targets="quality",
            model_type="regressor",
            evaluators=["default"],
        )

    last = mlflow.last_active_run()
    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": last.info.run_id,
        "run_name": last.info.run_name,
        "manual_metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "evaluate_metrics": {k: float(v) for k, v in eval_result.metrics.items()},
        "artifact_uri": artifact_uri,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Load a previously-trained pyfunc model and predict on the given rows."""
    try:
        model_uri = f"runs:/{req.run_id}/{ARTIFACT_PATH}"
        loaded = mlflow.pyfunc.load_model(model_uri)                       # NEW
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not loadable: {e}")

    df = pd.DataFrame(req.rows)
    preds = loaded.predict(df)
    return {
        "model_uri": model_uri,
        "n_rows": len(df),
        "predictions": list(preds),
    }
