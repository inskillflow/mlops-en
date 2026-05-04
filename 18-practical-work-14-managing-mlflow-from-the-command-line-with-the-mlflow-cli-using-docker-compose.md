<a id="top"></a>

# Chapter 18 — Today's topic: managing MLflow from the command line (`mlflow` CLI)

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Why a CLI when we have the UI?](#section-2) |
| 3 | [The new `cli` service in `docker-compose.yml`](#section-3) |
| 4 | [Project structure](#section-4) |
| 5 | [How to enter the CLI shell](#section-5) |
| 6 | [Generate sample data first](#section-6) |
| 7 | [`mlflow doctor` — sanity check](#section-7) |
| 8 | [`mlflow experiments` — manage experiments](#section-8) |
| 9 | [`mlflow runs` — manage runs](#section-9) |
| 10 | [`mlflow artifacts` — list / download / upload](#section-10) |
| 11 | [`mlflow models serve` — serve a registered model](#section-11) |
| 12 | [`mlflow db upgrade` — schema migration](#section-12) |
| 13 | [Cheat sheet](#section-13) |
| 14 | [Tear down](#section-14) |
| 15 | [Recap](#section-15) |

---

<a id="section-1"></a>

## 1. Objective

Today we don't add a new Python MLflow function. We add a **new way to interact with the same MLflow server**: the `mlflow` command-line interface. Same data, same backend, same Registry — different tool.

We'll add a small **`cli`** service to `docker-compose.yml`. It does nothing on its own — it just sits there with `mlflow` installed and `MLFLOW_TRACKING_URI` pointed at our server, ready for you to `exec` into.

> [!IMPORTANT]
> The CLI is what most CI/CD pipelines, batch jobs, and admin scripts actually use. The UI is for humans browsing; the CLI is for automation, scripting, and surgical operations.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Why a CLI when we have the UI?

| Use case | UI | CLI |
|---|---|---|
| Browse a few runs visually | best | painful |
| Bulk-delete 200 stale runs | painful | one-liner |
| Schedule nightly cleanup in cron / GitHub Actions | impossible | trivial |
| Download all artifacts of a run | several clicks | `mlflow artifacts download` |
| Apply a database migration after MLflow upgrade | impossible | `mlflow db upgrade` |
| Serve a Production model on a port for a smoke test | impossible | `mlflow models serve` |

The CLI is **the** scriptable interface — and it ships with the same `pip install mlflow`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. The new `cli` service in `docker-compose.yml`

We reuse the same `mlflow` + `fastapi` + `streamlit` services as Chapter 17 (so we already have experiments, runs and a registered model to play with) and we add **one** service:

```yaml
  cli:
    image: python:3.12-slim
    container_name: mlflow-cli
    working_dir: /workspace
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    command: >
      bash -c "pip install --no-cache-dir mlflow==2.16.2 &&
               echo 'mlflow CLI ready, run: docker compose exec cli bash' &&
               tail -f /dev/null"
    volumes:
      - ./cli-workspace:/workspace
    depends_on:
      mlflow:
        condition: service_healthy
    networks:
      - mlops-net
```

What this does:

- Builds **no** image — it's just `python:3.12-slim` with `pip install mlflow`.
- Sets `MLFLOW_TRACKING_URI` so every CLI command targets our server.
- `tail -f /dev/null` keeps it alive in the background.
- Mounts a local `./cli-workspace/` folder so files you download (`mlflow artifacts download …`) appear on your host.

> [!NOTE]
> We **don't** install `mlflow` in the existing `mlflow` container's PATH context to keep its image clean. A separate ephemeral CLI container is the cleaner pattern.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Project structure

```text
chap18-managing-mlflow-from-the-command-line/
├── docker-compose.yml          ← adds a `cli` service
├── cli-workspace/              ← bind mount for downloaded artifacts (created at first up)
├── mlflow/Dockerfile
├── fastapi/                    ← unchanged from chap17 (Registry endpoints)
└── streamlit/                  ← unchanged from chap17
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. How to enter the CLI shell

```bash
cd chap18-managing-mlflow-from-the-command-line
docker compose up -d --build
```

Wait until the cli container has finished its `pip install` (look at its logs):

```bash
docker compose logs -f cli
# look for: "mlflow CLI ready, run: docker compose exec cli bash"
```

Then enter the shell:

```bash
docker compose exec cli bash
```

You're now inside a Python image with the **same `mlflow` version** as the server, talking to it over the Docker network on `http://mlflow:5000`.

```bash
root@xxxxxxx:/workspace# mlflow --version
mlflow, version 2.16.2
root@xxxxxxx:/workspace# echo $MLFLOW_TRACKING_URI
http://mlflow:5000
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Generate sample data first

Before exploring the CLI, hit the FastAPI to create some experiments / runs / a registered model. From your **host**:

```bash
# Create 2 versions of WineQualityPredictor
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" -d "{\"alpha\":0.4,\"l1_ratio\":0.4}"
curl -X POST http://localhost:8000/train-and-register \
  -H "Content-Type: application/json" -d "{\"alpha\":0.7,\"l1_ratio\":0.3}"

# Promote v2 to Production
curl -X POST http://localhost:8000/promote \
  -H "Content-Type: application/json" \
  -d "{\"version\":2,\"stage\":\"Production\",\"archive_existing\":true}"
```

Now go back to the CLI shell.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. `mlflow doctor` — sanity check

```bash
mlflow doctor
```

```text
System information: Linux #1 SMP Debian ... x86_64
Python version: 3.12.x
MLflow version: 2.16.2
MLflow module location: /usr/local/lib/python3.12/...
Tracking URI: http://mlflow:5000
Registry URI: http://mlflow:5000
Active experiment ID: None
Active run ID: None
Active experiment artifact location: ...
MLflow environment variables:
  MLFLOW_TRACKING_URI: http://mlflow:5000
MLflow dependencies:
  ...
```

Hide secrets (very useful in shared CI logs):

```bash
mlflow doctor --mask-envs
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. `mlflow experiments` — manage experiments

List everything (active + deleted):

```bash
mlflow experiments search --view all
```

Create a new one:

```bash
mlflow experiments create --experiment-name cli_demo
# Created experiment 'cli_demo' with id 7
```

Rename it:

```bash
mlflow experiments rename --experiment-id 7 --new-name cli_demo_renamed
```

Soft-delete (the experiment goes to "deleted" state, can be restored):

```bash
mlflow experiments delete --experiment-id 7
mlflow experiments restore --experiment-id 7
```

Export a report as CSV (great for sharing with a non-technical stakeholder):

```bash
mlflow experiments csv --experiment-id 7 --filename /workspace/cli_demo.csv
ls /workspace/   # the file is also visible from your host in ./cli-workspace/
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. `mlflow runs` — manage runs

List all runs of an experiment:

```bash
mlflow runs list --experiment-id 1
```

Inspect a single run (full details, including all params/metrics/tags):

```bash
mlflow runs describe --run-id <RUN_ID>
```

Delete & restore (soft):

```bash
mlflow runs delete --run-id <RUN_ID>
mlflow runs restore --run-id <RUN_ID>
```

> [!IMPORTANT]
> Both `experiments delete` and `runs delete` are **soft** by default. To physically remove from the SQLite DB and free disk space, you also need `mlflow gc`:
>
> ```bash
> mlflow gc --backend-store-uri sqlite:////mlflow/database/mlflow.db
> ```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. `mlflow artifacts` — list / download / upload

List the artifacts of a run:

```bash
mlflow artifacts list --run-id <RUN_ID>
```

Typical output for a chap17 run:

```text
[
  {"path": "wine_quality_pyfunc",        "is_dir": true},
  {"path": "wine_quality_pyfunc/MLmodel", "is_dir": false},
  ...
]
```

Download everything to your host (via the bind mount):

```bash
mlflow artifacts download --run-id <RUN_ID> --dst-path /workspace/dl
ls /workspace/dl
# host:    ./cli-workspace/dl/...
```

Upload a local folder back into the run as a sub-artifact (e.g. attach an evaluation report your CI just produced):

```bash
mkdir -p /workspace/extra && echo "hello" > /workspace/extra/note.txt
mlflow artifacts log-artifacts \
  --local-dir /workspace/extra \
  --run-id <RUN_ID> \
  --artifact-path post_hoc_notes
```

Refresh the MLflow UI on `http://localhost:5000` → run → Artifacts → `post_hoc_notes/note.txt` is there.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. `mlflow models serve` — serve a registered model

This is the **killer CLI feature**: ship the Production model behind an HTTP server in one command. Inside the cli container:

```bash
mlflow models serve \
  --model-uri "models:/WineQualityPredictor/Production" \
  --host 0.0.0.0 --port 5001 \
  --env-manager local
```

Why `--env-manager local`? Because we already have all the deps installed in the cli container; we don't want MLflow to recreate a Conda env. In production you'd typically use `--env-manager virtualenv` or build a Docker image with `mlflow models build-docker`.

To call it from your host, expose port 5001 by **adding** to the `cli` service:

```yaml
    ports:
      - "5001:5001"
```

Then:

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d "{\"dataframe_records\":[{\"fixed acidity\":7.4,\"volatile acidity\":0.7,\"citric acid\":0.0,\"residual sugar\":1.9,\"chlorides\":0.076,\"free sulfur dioxide\":11.0,\"total sulfur dioxide\":34.0,\"density\":0.9978,\"pH\":3.51,\"sulphates\":0.56,\"alcohol\":9.4}]}"
# {"predictions": [5.4]}
```

You just stood up an inference service for the Production model with **zero application code**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. `mlflow db upgrade` — schema migration

When you bump the MLflow version, the SQL schema may need a migration. The CLI does it for you:

```bash
mlflow db upgrade sqlite:////mlflow/database/mlflow.db
```

(Note the four slashes: `sqlite:///` + the absolute path `/mlflow/database/mlflow.db`.)

Run this **only when the server is stopped** to avoid concurrent writes:

```bash
docker compose stop mlflow
docker compose exec cli mlflow db upgrade sqlite:////mlflow/database/mlflow.db
docker compose start mlflow
```

> [!NOTE]
> We're connecting to the same SQLite file because the bind-mount/volume is shared. In a real deployment with Postgres/MySQL, you'd point at the network DSN: `mlflow db upgrade postgresql://user:pwd@db:5432/mlflow`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-13"></a>

## 13. Cheat sheet

| Command | Effect |
|---|---|
| `mlflow doctor [--mask-envs]` | Diagnose installation / connectivity |
| `mlflow experiments search --view all` | List all experiments (incl. deleted) |
| `mlflow experiments create --experiment-name X` | Create an experiment |
| `mlflow experiments rename --experiment-id N --new-name X` | Rename |
| `mlflow experiments delete --experiment-id N` | Soft-delete |
| `mlflow experiments restore --experiment-id N` | Restore |
| `mlflow experiments csv --experiment-id N --filename f.csv` | Export to CSV |
| `mlflow runs list --experiment-id N --view all` | List runs |
| `mlflow runs describe --run-id R` | Show all params/metrics/tags |
| `mlflow runs delete --run-id R` | Soft-delete a run |
| `mlflow runs restore --run-id R` | Restore a run |
| `mlflow artifacts list --run-id R` | List a run's artifacts |
| `mlflow artifacts download --run-id R --dst-path D` | Download artifacts |
| `mlflow artifacts log-artifacts --local-dir L --run-id R --artifact-path P` | Upload artifacts |
| `mlflow models serve --model-uri models:/X/Production --port P --env-manager local` | Serve a registered model |
| `mlflow models build-docker --model-uri models:/X/Production --name img` | Build a Docker image around it |
| `mlflow db upgrade <DSN>` | Apply DB schema migration |
| `mlflow gc --backend-store-uri <DSN>` | Permanently delete soft-deleted runs |

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-14"></a>

## 14. Tear down

```bash
docker compose down              # keep volumes (and registered models)
docker compose down -v           # wipe everything including the SQLite DB and artifacts
rm -rf cli-workspace             # if you want to also clean local downloads
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-15"></a>

## 15. Recap

- The MLflow CLI ships with `mlflow` itself; pointing it at the server is just `MLFLOW_TRACKING_URI=...`.
- We spawned an ephemeral `cli` service in Docker Compose to keep host pollution at zero.
- `experiments`, `runs`, `artifacts` cover 90 % of admin tasks.
- `mlflow models serve` turns any registered model into an HTTP service in one command.
- `mlflow db upgrade` and `mlflow gc` are the housekeeping commands you'll thank yourself for knowing.

> [!IMPORTANT]
> Congratulations — this concludes the **core MLflow + Docker Compose** progression of the course. Everything from chapter 6 onward composes into one production-grade workflow:
> `Compose stack` → `set_experiment` → `start_run` → `log_param/log_metric` → `log_artifact` → `set_tag` → loops & sweeps → `autolog` → `signature/input_example` → `pyfunc + Conda env` → `load_model + evaluate` → `Model Registry` → **`CLI ops`**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 18 — managing MLflow from the command line</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
