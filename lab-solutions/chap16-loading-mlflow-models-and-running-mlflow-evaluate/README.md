# chap16 - loading MLflow models and running `mlflow.evaluate`

The full lesson lives at [`../16-practical-work-12-loading-mlflow-models-and-running-mlflow-evaluate-using-docker-compose.md`](../16-practical-work-12-loading-mlflow-models-and-running-mlflow-evaluate-using-docker-compose.md).

## Quick run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Two endpoints

| Endpoint | What |
|---|---|
| `POST /train-and-evaluate` | Trains a pyfunc model + runs `mlflow.evaluate` against the test set |
| `POST /predict` | Loads a pyfunc model from `run_id` and predicts on rows |

```bash
# 1) train, get a run_id back
curl -X POST http://localhost:8000/train-and-evaluate \
  -H "Content-Type: application/json" \
  -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"

# 2) predict (replace <RUN_ID> with the value from step 1)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"<RUN_ID>\",\"rows\":[{\"fixed acidity\":7.4,\"volatile acidity\":0.7,\"citric acid\":0.0,\"residual sugar\":1.9,\"chlorides\":0.076,\"free sulfur dioxide\":11.0,\"total sulfur dioxide\":34.0,\"density\":0.9978,\"pH\":3.51,\"sulphates\":0.56,\"alcohol\":9.4}]}"
```

In MLflow UI you'll see the auto-generated `residuals.png`, `prediction_distribution.png`, `eval_results_table.json` plus all the auto-computed regression metrics.

## Tear down

```bash
docker compose down
```
