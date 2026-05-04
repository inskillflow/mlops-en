# Chapter 09 — `log_param()` + `log_metric()`

Today's topic: **recording hyperparameters and evaluation metrics** in each run, so the MLflow UI is no longer empty.

The full lesson lives at [`../09-practical-work-5-logging-mlflow-params-and-metrics-with-log-param-and-log-metric-using-docker-compose.md`](../09-practical-work-5-logging-mlflow-params-and-metrics-with-log-param-and-log-metric-using-docker-compose.md).

---

## Quick run

```bash
docker compose up --build
```

Trigger a few trainings with different alpha / l1_ratio combinations:

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.1,\"l1_ratio\":0.1}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"l1_ratio\":0.5}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.9,\"l1_ratio\":0.9}"
```

Open MLflow UI → experiment **`wine_quality_chap09`** → click on the runs and compare `rmse`, `mae`, `r2` across hyperparameter combinations.

## Tear down

```bash
docker compose down
```

## The lines that are new vs Chapter 08

In `fastapi/app/main.py`, inside the `with mlflow.start_run(...)` block:

```python
mlflow.log_param("alpha", req.alpha)         # NEW
mlflow.log_param("l1_ratio", req.l1_ratio)   # NEW
mlflow.log_metric("rmse", rmse)              # NEW
mlflow.log_metric("mae", mae)                # NEW
mlflow.log_metric("r2", r2)                  # NEW
```
