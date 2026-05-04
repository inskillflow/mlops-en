<a id="top"></a>

# Chapter 05a — Today's topic: a minimal Streamlit app, dockerized

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

This is the **first** of three "warm-up" chapters (05a, 05b, 05c). Each one runs **one technology in one Docker container** so you can see it in isolation before chapter 06 assembles all three into a single Docker Compose stack.

Today: **Streamlit only**. No FastAPI, no MLflow. Goal: prove that a Streamlit app can be packaged in a Docker image, started with `docker compose up`, and reached at http://localhost:8501.

> [!NOTE]
> Streamlit is the UI layer of the full stack you'll build later. It's pure Python, uses widgets like `st.button`, and re-runs the whole script on every interaction. We'll use it as the user-facing entry point of MLOps demos.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Prerequisite

You need **Docker Desktop** (Windows/macOS) or **Docker Engine + Compose v2** (Linux).

> [!IMPORTANT]
> If Docker isn't installed yet, jump to [chapter 06, sections 1 to 3](./06-practical-work-2-installing-docker-desktop-and-running-mlflow-fastapi-streamlit-with-docker-compose.md) — they walk you through the full install on Windows / macOS / Linux. Then come back here.

Sanity-check:

```bash
docker --version
docker compose version
```

Both should print a version. If they do, you're ready.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Project structure

```text
chap05a-streamlit-minimal-with-docker/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── app/
    └── app.py
```

A grand total of **5 files**. That's all it takes.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. The code

### 4.1 `app/app.py` — the Streamlit application

```python
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Streamlit Minimal - chap 05a", page_icon=":sparkles:")

st.title("Streamlit Minimal - chap 05a")
st.write("A minimal Streamlit app running inside a Docker container.")

st.header("Inputs")
name = st.text_input("Your name", value="World")
n = st.slider("Pick a number", 0, 100, 42)

st.header("Output")
if st.button("Greet me"):
    st.success(f"Hello, {name}! Your number is {n}.")
    st.metric(label="Squared", value=n * n, delta=n)

st.header("A tiny dataframe")
df = pd.DataFrame({"x": list(range(5)), "y": [v * v for v in range(5)]})
st.dataframe(df, use_container_width=True)
```

Five widgets total: `text_input`, `slider`, `button`, `metric`, `dataframe`. Plenty to demo Streamlit's main idioms.

### 4.2 `requirements.txt`

```text
streamlit==1.39.0
pandas==2.2.3
```

### 4.3 `Dockerfile`

```dockerfile
FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
```

Two important details:
- `--server.address=0.0.0.0` makes Streamlit listen on all interfaces of the container. Without this, it would only listen on `127.0.0.1` *inside* the container and your host wouldn't be able to reach it.
- `EXPOSE 8501` is documentation only. The real port mapping happens in `docker-compose.yml`.

### 4.4 `docker-compose.yml`

```yaml
services:
  streamlit:
    build: .
    image: mlops/streamlit-minimal:latest
    container_name: streamlit-minimal
    ports:
      - "8501:8501"
    restart: unless-stopped
```

A single service. No volumes, no network, no `depends_on`. Just the bare minimum to get a container running.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Run, test, tear down

```bash
cd chap05a-streamlit-minimal-with-docker
docker compose up --build
```

The first build takes about 30 – 60 s (downloading the base image and installing Streamlit). Subsequent runs are instant.

When you see in the logs:

```text
streamlit-minimal  |   You can now view your Streamlit app in your browser.
streamlit-minimal  |   URL: http://0.0.0.0:8501
```

open [http://localhost:8501](http://localhost:8501) in your browser.

Type your name, move the slider, click **Greet me**. The page rerenders, the metric appears, the dataframe is shown. You're driving Streamlit from a container.

> [!NOTE]
> **Mini exercise.** Edit `app/app.py` and add `st.line_chart(df.set_index("x"))` after the dataframe. Save. Streamlit auto-detects the change → click "Rerun" in the top right. (No need to rebuild the image — Streamlit serves the file from the layer that was copied in `docker build`. You'd need a rebuild only if you change `requirements.txt`. We'll see hot-reload via bind mounts in chapter 06.)

Tear down:

```bash
docker compose down
```

That removes the container. The image stays cached for next time.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Recap and next chapter

You just built and ran a 1-service Docker app. Same pattern you'll re-use everywhere:

```text
Dockerfile + requirements.txt + app/      → docker build
docker-compose.yml                          → docker compose up
```

Next: **[chapter 05b](./05b-practical-work-1b-installing-fastapi-and-running-a-calculator-api-with-docker.md)** — same skeleton, but with FastAPI + Pydantic + uvicorn, and we expose a small REST API (a calculator).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Chapter 05a — Streamlit minimal</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
