# chap13 - automating MLflow logging with `mlflow.sklearn.autolog()`

The full lesson lives at [`../13-practical-work-9-automating-mlflow-logging-with-sklearn-autolog-using-docker-compose.md`](../13-practical-work-9-automating-mlflow-logging-with-sklearn-autolog-using-docker-compose.md).

## Quick run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Trigger

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
```

Then in the MLflow UI (`wine_quality_chap13_autolog`):
- **Parameters** = all constructor args of `ElasticNet` (auto)
- **Metrics** = `training_*` (auto) + `test_*` (manual)
- **Artifacts** = a full `model/` folder (auto), ready to be served with `mlflow models serve`

## Tear down

```bash
docker compose down
```
