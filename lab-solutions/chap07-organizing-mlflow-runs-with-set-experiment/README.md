# Chapter 07 — `mlflow.set_experiment()`

Today's topic: **creating or selecting an MLflow experiment** to group runs together.

The full lesson lives at [`../07-practical-work-3-organizing-mlflow-runs-with-set-experiment-using-docker-compose.md`](../07-practical-work-3-organizing-mlflow-runs-with-set-experiment-using-docker-compose.md).

---

## Quick run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

Trigger a training:

```bash
curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
```

Then open the MLflow UI and look for the experiment **`wine_quality_chap07`**.

## Tear down

```bash
docker compose down
```

## The single new line vs Chapter 06

In `fastapi/app/main.py`:

```python
mlflow.set_experiment("wine_quality_chap07")   # NEW
```
