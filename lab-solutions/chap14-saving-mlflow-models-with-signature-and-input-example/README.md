# chap14 - saving MLflow models with signature and input_example

The full lesson lives at [`../14-practical-work-10-saving-mlflow-models-with-signature-and-input-example-using-docker-compose.md`](../14-practical-work-10-saving-mlflow-models-with-signature-and-input-example-using-docker-compose.md).

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

After training, in MLflow UI → run → tab **Artifacts** → folder `model/` you should see:

- `MLmodel` (with the `signature:` block)
- `model.pkl`
- `input_example.json` (5 sample rows)

You can serve the model directly:

```bash
mlflow models serve -m "runs:/<RUN_ID>/model" -p 1234 --no-conda
curl -X POST http://localhost:1234/invocations \
  -H "Content-Type: application/json" \
  -d @input_example.json
```

## Tear down

```bash
docker compose down
```
