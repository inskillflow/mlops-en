# Chapter 08 — `start_run(run_name=...)` + `last_active_run()`

Today's topic: **naming runs explicitly** and **querying the last finished run** from outside the `with` block.

The full lesson lives at [`../08-practical-work-4-naming-mlflow-runs-with-start-run-and-last-active-run-using-docker-compose.md`](../08-practical-work-4-naming-mlflow-runs-with-start-run-and-last-active-run-using-docker-compose.md).

---

## Quick run

```bash
docker compose up --build
```

Trigger a training with a custom run name:

```bash
curl -X POST http://localhost:8000/train ^
     -H "Content-Type: application/json" ^
     -d "{\"alpha\":0.7,\"l1_ratio\":0.3,\"run_name\":\"baseline_v1\"}"
```

Open MLflow UI → experiment **`wine_quality_chap08`** → you should see a run literally called `baseline_v1`.

## Tear down

```bash
docker compose down
```

## The lines that are new vs Chapter 07

In `fastapi/app/main.py`:

```python
with mlflow.start_run(run_name=run_name):    # NEW: run_name
    ...

last = mlflow.last_active_run()              # NEW: query the run after it ended
```
