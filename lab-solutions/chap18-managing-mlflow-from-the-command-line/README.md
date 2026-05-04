# chap18 - managing MLflow from the command line

The full lesson lives at [`../18-practical-work-14-managing-mlflow-from-the-command-line-with-the-mlflow-cli-using-docker-compose.md`](../18-practical-work-14-managing-mlflow-from-the-command-line-with-the-mlflow-cli-using-docker-compose.md).

## Quick start

```bash
docker compose up -d --build

# Wait for the cli container to finish its pip install:
docker compose logs -f cli   # look for "mlflow CLI ready"
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| CLI container | `docker compose exec cli bash` |

## Generate some data first

```bash
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" -d "{\"alpha\":0.7,\"l1_ratio\":0.3}"
curl -X POST http://localhost:8000/promote \
  -H "Content-Type: application/json" \
  -d "{\"version\":2,\"stage\":\"Production\",\"archive_existing\":true}"
```

## Then jump into the CLI

```bash
docker compose exec cli bash

# inside the container
mlflow doctor
mlflow experiments search --view all
mlflow runs list --experiment-id 1
mlflow artifacts list --run-id <RUN_ID>
mlflow artifacts download --run-id <RUN_ID> --dst-path /workspace/dl

mlflow models serve \
  --model-uri "models:/WineQualityPredictor/Production" \
  --host 0.0.0.0 --port 5001 \
  --env-manager local
```

The `cli-workspace/` folder is a bind-mount, so anything you save under `/workspace/...` in the container appears next to the project on your host.

## Tear down

```bash
docker compose down       # keep models and runs
docker compose down -v    # wipe everything
```
