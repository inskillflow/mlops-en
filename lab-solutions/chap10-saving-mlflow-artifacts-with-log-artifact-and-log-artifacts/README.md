# chap10 - saving MLflow artifacts with `log_artifact` and `log_artifacts`

The full lesson lives at [`../10-practical-work-6-saving-mlflow-artifacts-with-log-artifact-and-log-artifacts-using-docker-compose.md`](../10-practical-work-6-saving-mlflow-artifacts-with-log-artifact-and-log-artifacts-using-docker-compose.md).

## Quick run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Trigger a training

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
```

Then go to MLflow UI → experiment `wine_quality_chap10` → pick a run → tab **Artifacts**. You should see `predictions_plot.png`, `test_predictions.csv`, and the folder `all_outputs/`.

## Tear down

```bash
docker compose down
```

(Volumes `chap10-...-mlflow-db` and `chap10-...-mlflow-artifacts` are kept by default.)
