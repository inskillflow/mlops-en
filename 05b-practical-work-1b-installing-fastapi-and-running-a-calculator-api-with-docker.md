<a id="top"></a>

# Chapter 05b — Today's topic: a FastAPI calculator API, dockerized

## Table of Contents

| # | Section |
|---|---|
| 1 | [Objective](#section-1) |
| 2 | [Prerequisite](#section-2) |
| 3 | [Project structure](#section-3) |
| 4 | [The code](#section-4) |
| 5 | [Run, test, tear down](#section-5) |
| 6 | [Recap and next chapter](#section-6) |

---

<a id="section-1"></a>

## 1. Objective

Second warm-up chapter. Today: **FastAPI only**. No Streamlit, no MLflow. We expose a tiny **calculator REST API** with four endpoints — `/add`, `/sub`, `/mul`, `/div` — using `Pydantic` for input validation and `uvicorn` as the ASGI server.

This is exactly the role FastAPI will play in the full stack: a clean, typed HTTP layer between the UI (Streamlit, chap 05a) and the brain (MLflow + scikit-learn, chap 05c onward).

> [!NOTE]
> FastAPI gives you **OpenAPI / Swagger UI for free**. You'll be able to try every endpoint from a browser without writing a single line of `curl`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Prerequisite

Same as [chapter 05a](./05a-practical-work-1a-installing-streamlit-and-running-a-minimal-app-with-docker.md): a working Docker Desktop / Docker Engine.

> [!IMPORTANT]
> If Docker isn't installed yet, read [chapter 06, sections 1 to 3](./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md) first.

```bash
docker --version
docker compose version
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Project structure

```text
chap05b-fastapi-calculator-with-docker/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── app/
    └── main.py
```

Same five-file skeleton as chap 05a — only `app/` differs.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The code

### 4.1 `app/main.py` — the calculator API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="FastAPI Calculator - chap 05b",
    description="A minimal FastAPI app, dockerized.",
    version="1.0.0",
)


class Operands(BaseModel):
    a: float
    b: float


@app.get("/")
def root():
    return {
        "app": "calculator",
        "endpoints": ["/health", "/add", "/sub", "/mul", "/div"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add")
def add(op: Operands):
    return {"op": "add", "a": op.a, "b": op.b, "result": op.a + op.b}


@app.post("/sub")
def sub(op: Operands):
    return {"op": "sub", "a": op.a, "b": op.b, "result": op.a - op.b}


@app.post("/mul")
def mul(op: Operands):
    return {"op": "mul", "a": op.a, "b": op.b, "result": op.a * op.b}


@app.post("/div")
def div(op: Operands):
    if op.b == 0:
        raise HTTPException(status_code=400, detail="Division by zero")
    return {"op": "div", "a": op.a, "b": op.b, "result": op.a / op.b}
```

A few things to notice:
- **`Operands(BaseModel)`** is the Pydantic input schema. FastAPI uses it to: parse the JSON body, validate types, and document the endpoint in Swagger.
- `/div` raises `HTTPException(400, ...)` on division by zero. FastAPI turns it into a proper HTTP 400 response.
- No global state, no async — perfectly fine for a demo.

### 4.2 `requirements.txt`

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
```

### 4.3 `Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`app.main:app` means: in module `app.main`, take the variable named `app`. Same `--host 0.0.0.0` reasoning as chapter 05a (listen on all interfaces inside the container).

### 4.4 `docker-compose.yml`

```yaml
services:
  fastapi:
    build: .
    image: mlops/fastapi-calculator:latest
    container_name: fastapi-calculator
    ports:
      - "8000:8000"
    restart: unless-stopped
```

Single service. No volumes, no network, no `depends_on`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Run, test, tear down

```bash
cd chap05b-fastapi-calculator-with-docker
docker compose up --build
```

When you see:

```text
fastapi-calculator  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 5.1 Try it from the browser (Swagger UI)

Open [http://localhost:8000/docs](http://localhost:8000/docs).

You should see all five endpoints listed. Click **POST /add** → **Try it out** → enter:

```json
{ "a": 7, "b": 5 }
```

→ **Execute**. You get back:

```json
{ "op": "add", "a": 7.0, "b": 5.0, "result": 12.0 }
```

### 5.2 Try it from a terminal (curl)

```bash
# Health check
curl http://localhost:8000/health

# Add 7 + 5
curl -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d "{\"a\":7,\"b\":5}"

# Multiply 6 * 9
curl -X POST http://localhost:8000/mul \
  -H "Content-Type: application/json" \
  -d "{\"a\":6,\"b\":9}"

# Divide by zero on purpose - returns HTTP 400
curl -i -X POST http://localhost:8000/div \
  -H "Content-Type: application/json" \
  -d "{\"a\":1,\"b\":0}"
```

### 5.3 Mini exercise

Add a fifth operation `POST /pow` that returns `{"result": op.a ** op.b}`. Save `app/main.py`, then rebuild:

```bash
docker compose up --build
```

Refresh `/docs` — your new endpoint shows up automatically. That's the FastAPI superpower: code IS the documentation.

### 5.4 Tear down

```bash
docker compose down
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Recap and next chapter

You now have the **same skeleton** as chap 05a (Dockerfile + requirements + app + compose) but with FastAPI instead of Streamlit. That's the whole point: every service in a Dockerized app fits this 5-file mold.

Next: **[chapter 05c](./05c-practical-work-1c-installing-mlflow-server-with-docker-and-logging-a-first-run.md)** — same skeleton, but with the MLflow server. After that, chapter 06 finally assembles the three into one multi-service `docker-compose.yml`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 05b — FastAPI calculator</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
