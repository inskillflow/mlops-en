# chap15 - wrapping sklearn with `mlflow.pyfunc` + custom Conda env

The full lesson lives at [`../15-practical-work-11-wrapping-sklearn-models-with-mlflow-pyfunc-and-custom-conda-env-using-docker-compose.md`](../15-practical-work-11-wrapping-sklearn-models-with-mlflow-pyfunc-and-custom-conda-env-using-docker-compose.md).

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
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"
```

After training, in MLflow UI → run → **Artifacts** → folder `wine_quality_pyfunc/`:

```
wine_quality_pyfunc/
├── MLmodel              (flavor: python_function only)
├── conda.yaml           (custom env: mlflow, sklearn, cloudpickle, joblib...)
├── python_model.pkl     (cloudpickle of WineQualityWrapper)
├── artifacts/
│   └── sklearn_model.pkl
├── input_example.json
└── requirements.txt
```

The wrapper applies two business rules: clip predictions to [3, 9] and round to one decimal.

## Tear down

```bash
docker compose down
```
