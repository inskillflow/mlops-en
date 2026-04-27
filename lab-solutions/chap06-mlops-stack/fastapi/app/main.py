import os
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(title="MLOps API", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "FastAPI is running", "mlflow": MLFLOW_URI}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/log-run")
def log_run(req: PredictRequest):
    mlflow.set_experiment("fastapi-demo")
    with mlflow.start_run():
        mlflow.log_param("n_features", len(req.features))
        mlflow.log_metric("sum", float(sum(req.features)))
        mlflow.set_tag("source", "fastapi")
    return {"logged": True, "features": req.features}
