# chap11 - tagging MLflow runs with `set_tag` / `set_tags`

The full lesson lives at [`../11-practical-work-7-tagging-mlflow-runs-with-set-tag-and-set-tags-using-docker-compose.md`](../11-practical-work-7-tagging-mlflow-runs-with-set-tag-and-set-tags-using-docker-compose.md).

## Quick run

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Trigger a few runs with different `triggered_by`

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.3,\"triggered_by\":\"streamlit-ui\"}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.5,\"triggered_by\":\"curl-cli\"}"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.7,\"triggered_by\":\"scheduled-job\"}"
```

Then in MLflow UI search bar, type:

```text
tags.triggered_by = "scheduled-job"
```

You should see only 1 run.

## Tear down

```bash
docker compose down
```
