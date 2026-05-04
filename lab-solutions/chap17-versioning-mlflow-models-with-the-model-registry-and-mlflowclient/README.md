# chap17 - versioning MLflow models with the Model Registry and `MlflowClient`

The full lesson lives at [`../17-practical-work-13-versioning-mlflow-models-with-the-model-registry-and-mlflowclient-using-docker-compose.md`](../17-practical-work-13-versioning-mlflow-models-with-the-model-registry-and-mlflowclient-using-docker-compose.md).

## Quick run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| MLflow UI (Experiments + **Models** tabs) | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /train-and-register` | Train, log + `mlflow.register_model(...)` under `WineQualityPredictor` |
| `GET /versions` | List all versions of `WineQualityPredictor` |
| `POST /promote` | `MlflowClient.transition_model_version_stage(...)` |
| `POST /predict-production` | Load `models:/WineQualityPredictor/Production` and predict |

## End-to-end demo

```bash
# v1
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"

# v2
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" -d "{\"alpha\":0.7,\"l1_ratio\":0.3}"

# Promote v2 to Production (archive v1 if it was there)
curl -X POST http://localhost:8000/promote \
  -H "Content-Type: application/json" \
  -d "{\"version\":2,\"stage\":\"Production\",\"archive_existing\":true}"

# Consumer never sees a run_id
curl -X POST http://localhost:8000/predict-production \
  -H "Content-Type: application/json" \
  -d "{\"rows\":[{\"fixed acidity\":7.4,\"volatile acidity\":0.7,\"citric acid\":0.0,\"residual sugar\":1.9,\"chlorides\":0.076,\"free sulfur dioxide\":11.0,\"total sulfur dioxide\":34.0,\"density\":0.9978,\"pH\":3.51,\"sulphates\":0.56,\"alcohol\":9.4}]}"
```

In the MLflow UI go to the **Models** tab to see the lifecycle of `WineQualityPredictor`.

## Tear down

```bash
docker compose down       # keep registered models
docker compose down -v    # wipe everything
```
