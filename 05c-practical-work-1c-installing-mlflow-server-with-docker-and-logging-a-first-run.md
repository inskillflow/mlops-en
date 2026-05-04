<a id="top"></a>

# Chapter 05c — Today's topic: an MLflow server in Docker + a first logged run

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Prerequisite](#section-2) |
| 3 | [Project structure](#section-3) |
| 4 | [The code](#section-4) |
| 5 | [Run, log a first run, tear down](#section-5) |
| 6 | [Recap and next chapter](#section-6) |

---

<a id="section-1"></a>

## 1. Objective

Last warm-up chapter. Today: **MLflow only**. No Streamlit, no FastAPI. We package the MLflow tracking server in a Docker container and we log our very first run from the host.

> [!NOTE]
> **Intentional duplication with chapter 05.** Chapter 05 installed MLflow inside a venv on Ubuntu. This chapter installs **the same MLflow** but inside a Docker image — so it works identically on Windows, macOS and Linux without touching your system Python. Same goal, different packaging. Choose either path; from chapter 06 onward we use the Docker version everywhere.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Prerequisite

Same as 05a / 05b: a working Docker Desktop / Docker Engine.

> [!IMPORTANT]
> If Docker isn't installed yet, read [chapter 06, sections 1 to 3](./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md) first.

You also need **a small host-side Python venv** to run the `hello_mlflow.py` script that talks to the server. If you completed chapter 05, just reuse that venv. Otherwise:

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install mlflow==2.16.2
```

```powershell
# Windows PowerShell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install mlflow==2.16.2
```

The same MLflow version on host **and** in the container — that's important.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Project structure

```text
chap05c-mlflow-minimal-with-docker/
├── README.md
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
└── hello_mlflow.py        ← runs on the HOST against the dockerized server
```

Same Docker single-service skeleton as 05a / 05b, plus a small Python script that runs **on the host** to log the first run.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The code

### 4.1 `mlflow/Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /mlflow
RUN pip install --no-cache-dir mlflow==2.16.2
EXPOSE 5000
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///database/mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", "--port", "5000"]
```

Two storage knobs to know:
- `--backend-store-uri sqlite:///database/mlflow.db` — experiments / runs / params / metrics live in a SQLite file.
- `--default-artifact-root /mlflow/mlruns` — large blobs (models, plots, CSV) live in a folder.

We mount **both** as volumes so they survive `docker compose down`.

### 4.2 `docker-compose.yml`

```yaml
services:
  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow-minimal:latest
    container_name: mlflow-minimal
    ports:
      - "5000:5000"
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
    restart: unless-stopped

volumes:
  mlflow-db:
  mlflow-artifacts:
```

> [!IMPORTANT]
> Without the **two volumes**, every `docker compose down` would erase all your runs. With them, even `docker compose down` (without `-v`) keeps your tracking history intact across restarts.

### 4.3 `hello_mlflow.py` — the first run, executed from the host

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("hello_mlflow")

with mlflow.start_run(run_name="my_first_run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 5)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.18)

print("Done. Open http://localhost:5000 to see your run.")
```

Five lines that matter:
- `set_tracking_uri(...)` — point at the dockerized server. From the host, that's `localhost:5000`. From another container, it would be `http://mlflow:5000` (you'll see this in chap 06).
- `set_experiment(...)` — group runs under a named experiment.
- `start_run(...)` — open a run; everything inside the `with` block is attached to it.
- `log_param(...)` and `log_metric(...)` — the two MLflow building blocks. They have a whole chapter each later (chap 09).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Run, log a first run, tear down

### 5.1 Start the MLflow server

```bash
cd chap05c-mlflow-minimal-with-docker
docker compose up --build
```

When you see:

```text
mlflow-minimal  | [INFO] Listening at: http://0.0.0.0:5000
```

open [http://localhost:5000](http://localhost:5000). You should see the empty MLflow UI, "**No experiments yet.**".

### 5.2 Log your first run from the host

In **another terminal** (keep the server running in the first one):

```bash
# activate your host venv first (see Section 2)
python hello_mlflow.py
```

Output:

```text
Done. Open http://localhost:5000 to see your run.
```

Refresh the MLflow UI. The experiment **`hello_mlflow`** now appears, with one run **`my_first_run`**, two params, two metrics. Click the run to inspect.

### 5.3 Mini exercise

Re-run `python hello_mlflow.py` two more times, but each time edit the values:

```python
mlflow.log_param("learning_rate", 0.001)   # then 0.1
mlflow.log_metric("accuracy", 0.95)        # then 0.88
```

You'll have **three runs** in the experiment. In the UI, click the column-header sort on `metrics.accuracy` to find the best run. Welcome to experiment tracking.

### 5.4 Tear down

```bash
# Keep all your runs (volumes survive)
docker compose down

# Or wipe everything (volumes deleted)
docker compose down -v
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Recap and next chapter

You've now seen each technology of the stack **in isolation**:

| Chapter | Service | Port |
|---|---|---|
| [05a](./05a-practical-work-1a-installing-streamlit-and-running-a-minimal-app-with-docker.md) | Streamlit | 8501 |
| [05b](./05b-practical-work-1b-installing-fastapi-and-running-a-calculator-api-with-docker.md) | FastAPI | 8000 |
| **05c** (this one) | **MLflow** | **5000** |

Next: **[chapter 06](./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md)** assembles the three into one multi-service Docker Compose stack — Streamlit calls FastAPI, FastAPI logs to MLflow — and explains the Docker fundamentals (images, networks, volumes, healthchecks) in depth.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 05c — MLflow server minimal</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
