# chap12 - multiple runs & multiple experiments

The full lesson lives at [`../12-practical-work-8-running-multiple-mlflow-runs-and-experiments-with-loops-using-docker-compose.md`](../12-practical-work-8-running-multiple-mlflow-runs-and-experiments-with-loops-using-docker-compose.md).

## Quick run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Two batch endpoints

| Endpoint | What it does |
|---|---|
| `POST /train-grid` | 9 ElasticNet runs (3 alphas × 3 l1_ratios) in **one** experiment `wine_quality_chap12_grid` |
| `POST /train-sweep` | 3 alphas × 3 families (ElasticNet, Ridge, Lasso) = **9 runs** across **3 experiments** |

```bash
curl -X POST http://localhost:8000/train-grid
curl -X POST http://localhost:8000/train-sweep
```

After both, you should see **4 experiments** in the MLflow UI and a total of 18 runs.

## Tear down

```bash
docker compose down
```
