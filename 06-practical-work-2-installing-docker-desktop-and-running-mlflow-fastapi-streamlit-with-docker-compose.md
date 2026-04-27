<a id="top"></a>

# Practical Work 2 — Installing Docker Desktop and Running MLflow + FastAPI + Streamlit with Docker Compose

## Table of Contents

| # | Section |
|---|---|
| 1 | [Introduction](#section-1) |
| 2 | [Prerequisites](#section-2) |
| 3 | [What Is Docker?](#section-3) |
| 4 | [Docker Core Concepts — Image, Container, Volume, Network](#section-4) |
| 5 | [Installing Docker Desktop](#section-5) |
| 6 | [Verifying the Installation](#section-6) |
| 7 | [Basic Docker Commands](#section-7) |
| 8 | [What Is Docker Compose?](#section-8) |
| 9 | [Project Structure for the Stack](#section-9) |
| 10 | [The MLflow Service](#section-10) |
| 11 | [The FastAPI Service](#section-11) |
| 12 | [The Streamlit Service](#section-12) |
| 13 | [The Complete `docker-compose.yml`](#section-13) |
| 14 | [Running the Full Stack](#section-14) |
| 15 | [Volumes, Persistence, and Networking](#section-15) |
| 16 | [Troubleshooting](#section-16) |
| 17 | [Conclusion](#section-17) |
| 18 | [Appendix 00 — Useful Docker and Compose Commands](#appendix-00) |
| 19 | [Appendix 01 — `Dockerfile` vs `docker-compose.yml`](#appendix-01) |
| 20 | [Appendix 02 — Cleaning Up Docker (Disk Space)](#appendix-02) |
| 21 | [Appendix 03 — Hot Reload for Development](#appendix-03) |
| 22 | [Appendix 04 — WSL 2 Backend on Windows](#appendix-04) |
| 23 | [Appendix 05 — Going to Production (PostgreSQL + S3)](#appendix-05) |

---

<a id="section-1"></a>

## 1. Introduction

This practical work explains how to **containerize** an MLOps stack with **Docker** and orchestrate it with **Docker Compose**.

The objective is to:

- understand what Docker is and why it is used in MLOps;
- install **Docker Desktop** on your machine;
- learn the core Docker vocabulary: **image**, **container**, **volume**, **network**;
- build and run three services together with a single command:
  - **MLflow** — the tracking server;
  - **FastAPI** — a REST API that serves a model;
  - **Streamlit** — a small web UI that calls the API;
- configure **volumes** so that MLflow data survives container restarts;
- expose each service on its own port from the host machine.

> [!IMPORTANT]
> This is a continuation of **Practical Work 1**, where MLflow was installed in a Python virtual environment on Ubuntu. Here, we replace the virtual environment with **containers** so the stack runs the same way on any machine that has Docker.

> [!NOTE]
> The same `docker-compose.yml` works on Windows, macOS, and Linux as long as Docker Desktop (or the Docker Engine on Linux) is installed.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Prerequisites

You need:

- a machine with **at least 8 GB of RAM** (16 GB recommended);
- around **10 GB of free disk space** for images and volumes;
- administrator / `sudo` privileges to install Docker Desktop;
- on **Windows 10/11**: virtualization enabled in BIOS and **WSL 2** available;
- on **macOS**: macOS 12 or newer (Intel or Apple Silicon);
- on **Linux**: a recent distribution such as Ubuntu 22.04 / 24.04;
- a working internet connection (Docker Hub will be used to pull images);
- a basic terminal (PowerShell, bash, or zsh).

> [!IMPORTANT]
> No Python installation is required on the host machine for this practical work. Python runs **inside** the containers.

> [!WARNING]
> Hyper-V, WSL 2, and some VPN clients can interfere with Docker networking. If something does not work, see **Section 16 — Troubleshooting** and **Appendix 04 — WSL 2 Backend on Windows**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. What Is Docker?

**Docker** is a platform that packages an application and **all its dependencies** (Python version, system libraries, configuration files, etc.) into a single portable unit called a **container**.

A container behaves like a very small, isolated machine. It contains:

- the application code;
- the runtime (for example Python 3.12);
- the operating system libraries the application needs;
- nothing else — it does **not** boot a full operating system.

### 3.1 Why Docker for MLOps?

Machine learning projects suffer from one classic problem:

> *"It works on my machine."*

Docker removes that problem because:

- the **same image** runs the same way on every machine;
- the **same versions** of Python and libraries are guaranteed;
- the **same configuration** is shipped with the code;
- a teammate, a CI server, or a production server can reproduce the run with one command.

> [!IMPORTANT]
> In MLOps, **reproducibility** is one of the main reasons to use Docker. An MLflow run that was logged from a container can be re-executed from the same image, months later, without rebuilding the environment by hand.

### 3.2 Docker vs virtual machine

| Aspect | Virtual machine | Docker container |
| --- | --- | --- |
| Runs a full OS | Yes (full kernel) | No (shares the host kernel) |
| Boot time | Minutes | Seconds (or less) |
| Disk size | Several GB | Often a few hundred MB |
| Isolation | Strong | Strong enough for most apps |
| Portable image | OVF / VMDK | Docker image (small, layered) |

> [!NOTE]
> Containers are **not** virtual machines. They are isolated processes that share the host's Linux kernel.

### 3.3 What is Docker Desktop?

**Docker Desktop** is the official Docker application for **Windows** and **macOS**. It bundles together:

- the **Docker Engine** (the daemon that runs containers);
- the **Docker CLI** (the `docker` command);
- **Docker Compose** (the `docker compose` command);
- a **graphical interface** to inspect containers, images, volumes, and logs;
- on Windows: integration with **WSL 2** so containers run inside a Linux virtual machine managed for you.

> [!NOTE]
> On Linux, Docker Desktop is optional. Most Linux users install the **Docker Engine** directly and use the CLI. Docker Desktop on Linux is mainly useful for the GUI.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Docker Core Concepts — Image, Container, Volume, Network

Before writing the `docker-compose.yml`, four words must be perfectly clear.

### 4.1 Image

An **image** is a read-only template that contains everything needed to run an application: OS layers, Python, your code, and your dependencies.

- An image is built from a **Dockerfile**.
- An image has a **name** and a **tag**, for example `python:3.12-slim` or `mlflow-server:latest`.
- An image is **immutable**: once built, it does not change.

> [!NOTE]
> You can think of an image as a **class** in object-oriented programming.

### 4.2 Container

A **container** is a running **instance** of an image.

- You can start, stop, and remove containers.
- Several containers can run from the **same image** at the same time.
- A container has its own filesystem, network interface, and process space.

> [!NOTE]
> You can think of a container as an **object** (an instance of a class).

### 4.3 Volume

A **volume** is a folder managed by Docker and mounted inside a container so that **data survives** when the container is removed or recreated.

Without a volume:

- the SQLite file `mlflow.db` lives inside the container;
- if the container is deleted, the database is **gone**.

With a volume:

- the SQLite file is stored in a Docker-managed folder on the host;
- the container can be deleted and recreated, the database is still there.

> [!IMPORTANT]
> For MLflow, **always use volumes** to store the backend database and the artifacts. Otherwise every `docker compose down` would erase the experiment history.

### 4.4 Network

A **network** is a virtual network that connects containers together.

- Containers on the same Docker network can reach each other by **service name** (DNS).
- For example, FastAPI can reach MLflow with the URL `http://mlflow:5000`.
- The host machine reaches a container through a **published port** such as `8000:8000`.

> [!NOTE]
> Inside the network: use the **service name**.
> From the host browser: use `http://localhost:<port>`.

### 4.5 Summary diagram

```text
+----------------------------------------------------------------+
|                       Host machine                             |
|                                                                |
|   Browser  ->  http://localhost:8501  (Streamlit UI)           |
|   Browser  ->  http://localhost:8000  (FastAPI docs)           |
|   Browser  ->  http://localhost:5000  (MLflow UI)              |
|                                                                |
|   +--------------------------------------------------------+   |
|   |               Docker network: mlops-net                |   |
|   |                                                        |   |
|   |   [streamlit] ---http://fastapi:8000---> [fastapi]     |   |
|   |                                              |         |   |
|   |                                              v         |   |
|   |                                  http://mlflow:5000    |   |
|   |                                              |         |   |
|   |                                              v         |   |
|   |                                          [mlflow]      |   |
|   |                                              |         |   |
|   |                                              v         |   |
|   |                                  Volume: mlflow-data   |   |
|   +--------------------------------------------------------+   |
+----------------------------------------------------------------+
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Installing Docker Desktop

### 5.1 Installation on Windows 10 / 11

**Step 1 — Enable WSL 2**

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

Then restart the computer.

> [!NOTE]
> `wsl --install` installs WSL 2 and Ubuntu by default. If WSL is already installed, this command does nothing harmful.

**Step 2 — Download Docker Desktop**

Go to:

```text
https://www.docker.com/products/docker-desktop/
```

Download **Docker Desktop for Windows** (the `.exe` installer).

**Step 3 — Run the installer**

- Double-click the `.exe`.
- Keep the option **"Use WSL 2 instead of Hyper-V"** checked.
- Finish the installation and **restart** if asked.

**Step 4 — Launch Docker Desktop**

- Open **Docker Desktop** from the Start menu.
- Wait until the bottom-left status indicator becomes **green** ("Engine running").
- Accept the license agreement.

> [!IMPORTANT]
> Docker Desktop must be **running** in the background for the `docker` and `docker compose` commands to work.

> [!WARNING]
> If you see *"WSL 2 installation is incomplete"*, see **Appendix 04**.

---

### 5.2 Installation on macOS

**Step 1 — Download Docker Desktop**

Go to the same page:

```text
https://www.docker.com/products/docker-desktop/
```

Choose the right installer:

- **Apple Silicon** (M1, M2, M3, M4) — the `arm64` `.dmg`;
- **Intel** — the `amd64` `.dmg`.

**Step 2 — Install**

- Open the `.dmg`.
- Drag **Docker** into the **Applications** folder.
- Launch **Docker** from Launchpad.
- Approve the privileged helper installation when prompted.

**Step 3 — Verify the menu bar**

A whale icon should appear in the menu bar. When it stops animating, Docker is ready.

---

### 5.3 Installation on Ubuntu (Docker Engine, no Desktop)

On Ubuntu, the recommended setup for this practical work is the **Docker Engine** (CLI only).

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Allow your user to run `docker` without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

> [!NOTE]
> After `usermod`, you may need to log out and log back in for the group change to take effect.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Verifying the Installation

Open a terminal (PowerShell on Windows, Terminal on macOS, bash on Linux) and run:

```bash
docker --version
docker compose version
docker run hello-world
```

You should see:

- a Docker version line;
- a Docker Compose version line;
- a friendly "Hello from Docker!" message produced by the `hello-world` container.

> [!IMPORTANT]
> If `docker run hello-world` works, your installation is complete. You are ready for the rest of this practical work.

> [!WARNING]
> If you get *"Cannot connect to the Docker daemon"*, Docker Desktop is not running (Windows / macOS) or the `docker` service is not started (Linux: `sudo systemctl start docker`).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Basic Docker Commands

These commands are useful even without Compose.

```bash
docker images              # list local images
docker ps                  # list running containers
docker ps -a               # list all containers (including stopped)
docker pull python:3.12-slim
docker run -it --rm python:3.12-slim python -c "print('hi')"
docker stop <container_id>
docker rm <container_id>
docker rmi <image_id>
docker logs <container_id>
docker exec -it <container_id> bash
```

> [!NOTE]
> The flag `--rm` removes the container automatically when it exits. It is very useful for one-shot commands.

> [!IMPORTANT]
> Tag and ID are interchangeable in most commands. You can use the first 3–4 characters of an ID (it is auto-completed).

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. What Is Docker Compose?

Running three containers by hand with `docker run` is painful. You would have to:

- create a network manually;
- start each container with the right environment variables;
- mount the right volumes for each service;
- remember the start order (MLflow before FastAPI, etc.).

**Docker Compose** solves all of that. It reads a single YAML file (`docker-compose.yml`) and starts the **whole stack** with one command:

```bash
docker compose up
```

A Compose file describes **services** (containers), **volumes** (persistent storage), and **networks** (how services talk to each other).

> [!IMPORTANT]
> One `docker-compose.yml` = one full environment.
> You can hand it to a teammate, and the entire MLOps stack runs in seconds on their machine.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Project Structure for the Stack

We will create the following folder structure on the host machine:

```text
mlops-stack/
├── docker-compose.yml
├── mlflow/
│   └── Dockerfile
├── fastapi/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── main.py
└── streamlit/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        └── app.py
```

Create it now:

```bash
mkdir mlops-stack
cd mlops-stack
mkdir mlflow fastapi streamlit
mkdir fastapi/app streamlit/app
```

> [!NOTE]
> Each service has its own folder, its own `Dockerfile`, and its own `requirements.txt`. This keeps the responsibilities clean and the images small.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. The MLflow Service

### 10.1 `mlflow/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow==2.16.2

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///database/mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
```

> [!IMPORTANT]
> The host **must** be `0.0.0.0` inside a container. With `127.0.0.1`, MLflow would only listen on the container's loopback interface and other containers could not reach it.

> [!NOTE]
> The folders `database/` and `mlruns/` will be **mounted from a Docker volume**. They do not need to exist in the image because Compose will create them at runtime.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. The FastAPI Service

### 11.1 `fastapi/requirements.txt`

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
mlflow==2.16.2
scikit-learn==1.5.2
numpy==2.1.1
```

### 11.2 `fastapi/app/main.py`

```python
import os
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(title="MLOps API", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "FastAPI is running", "mlflow": MLFLOW_URI}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/log-run")
def log_run(req: PredictRequest):
    mlflow.set_experiment("fastapi-demo")
    with mlflow.start_run():
        mlflow.log_param("n_features", len(req.features))
        mlflow.log_metric("sum", float(sum(req.features)))
        mlflow.set_tag("source", "fastapi")
    return {"logged": True, "features": req.features}
```

### 11.3 `fastapi/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> [!IMPORTANT]
> `MLFLOW_TRACKING_URI` is read from an environment variable that Compose will inject. The default `http://mlflow:5000` uses the **service name** `mlflow` as a hostname, which only works inside the Docker network.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-12"></a>

## 12. The Streamlit Service

### 12.1 `streamlit/requirements.txt`

```text
streamlit==1.39.0
requests==2.32.3
```

### 12.2 `streamlit/app/app.py`

```python
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://fastapi:8000")

st.title("MLOps Demo — Streamlit + FastAPI + MLflow")

st.write(f"API URL: `{API_URL}`")

features_text = st.text_input("Features (comma separated)", "1.0, 2.0, 3.0")

if st.button("Send to FastAPI"):
    try:
        features = [float(x.strip()) for x in features_text.split(",")]
        r = requests.post(f"{API_URL}/log-run", json={"features": features}, timeout=10)
        st.success("Logged!")
        st.json(r.json())
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown("**MLflow UI** — open [http://localhost:5000](http://localhost:5000)")
```

### 12.3 `streamlit/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
```

> [!NOTE]
> Streamlit must also listen on `0.0.0.0` so the host browser can reach it through the published port `8501`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-13"></a>

## 13. The Complete `docker-compose.yml`

Create the file `mlops-stack/docker-compose.yml`:

```yaml
services:

  mlflow:
    build:
      context: ./mlflow
    image: mlops/mlflow:latest
    container_name: mlflow
    ports:
      - "5000:5000"
    volumes:
      - mlflow-db:/mlflow/database
      - mlflow-artifacts:/mlflow/mlruns
    networks:
      - mlops-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys;urllib.request.urlopen('http://localhost:5000/').read();"]
      interval: 10s
      timeout: 5s
      retries: 5

  fastapi:
    build:
      context: ./fastapi
    image: mlops/fastapi:latest
    container_name: fastapi
    ports:
      - "8000:8000"
    environment:
      MLFLOW_TRACKING_URI: "http://mlflow:5000"
    depends_on:
      mlflow:
        condition: service_healthy
    networks:
      - mlops-net
    restart: unless-stopped

  streamlit:
    build:
      context: ./streamlit
    image: mlops/streamlit:latest
    container_name: streamlit
    ports:
      - "8501:8501"
    environment:
      API_URL: "http://fastapi:8000"
    depends_on:
      - fastapi
    networks:
      - mlops-net
    restart: unless-stopped

volumes:
  mlflow-db:
  mlflow-artifacts:

networks:
  mlops-net:
    driver: bridge
```

> [!IMPORTANT]
> Three services, two named volumes, one network — and the whole stack starts in the right order thanks to `depends_on` and the MLflow `healthcheck`.

> [!NOTE]
> The `image:` line under each `build:` block sets a friendly name like `mlops/mlflow:latest`. Without it, Compose would pick a long auto-generated name.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-14"></a>

## 14. Running the Full Stack

From the `mlops-stack/` folder:

```bash
docker compose up --build
```

What happens:

1. Compose **builds** the three images (first run only, or after a code change).
2. Compose creates the network `mlops-net` and the two volumes.
3. MLflow starts. Compose waits for its healthcheck to pass.
4. FastAPI starts and reaches MLflow with `http://mlflow:5000`.
5. Streamlit starts and reaches FastAPI with `http://fastapi:8000`.

Open in the browser:

| Service | URL |
| --- | --- |
| Streamlit UI | [http://localhost:8501](http://localhost:8501) |
| FastAPI docs | [http://localhost:8000/docs](http://localhost:8000/docs) |
| MLflow UI | [http://localhost:5000](http://localhost:5000) |

To stop the stack (Ctrl+C, then):

```bash
docker compose down
```

To stop **and** delete the volumes (full reset):

```bash
docker compose down -v
```

> [!WARNING]
> `docker compose down -v` deletes the MLflow database and all artifacts. Use it only when you really want a clean slate.

To run the stack in the background (detached):

```bash
docker compose up -d --build
docker compose logs -f
```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-15"></a>

## 15. Volumes, Persistence, and Networking

### 15.1 Two named volumes

| Volume | Mounted at | Purpose |
| --- | --- | --- |
| `mlflow-db` | `/mlflow/database` | SQLite file `mlflow.db` (experiments, runs, params, metrics) |
| `mlflow-artifacts` | `/mlflow/mlruns` | Artifacts (models, plots, files attached to runs) |

Inspect the volumes:

```bash
docker volume ls
docker volume inspect mlops-stack_mlflow-db
```

> [!IMPORTANT]
> Volume names are prefixed with the project name (the folder containing `docker-compose.yml`). Here it is `mlops-stack_mlflow-db`.

### 15.2 Service-name networking

Inside the `mlops-net` network:

- FastAPI reaches MLflow at `http://mlflow:5000` — **not** `localhost:5000`;
- Streamlit reaches FastAPI at `http://fastapi:8000` — **not** `localhost:8000`.

> [!NOTE]
> `localhost` inside a container points to **the container itself**, not to the host or to other containers. Always use the **service name** for inter-container calls.

### 15.3 Published ports

| Service | Container port | Host port |
| --- | --- | --- |
| MLflow | 5000 | 5000 |
| FastAPI | 8000 | 8000 |
| Streamlit | 8501 | 8501 |

The mapping `"8000:8000"` means: *traffic on host port 8000 is forwarded to container port 8000*. You may change the **left** number freely if a port is already used on your machine, for example `"18000:8000"`.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-16"></a>

## 16. Troubleshooting

### Problem — `docker: command not found`

Docker Desktop is not installed, or its CLI is not on your `PATH`. Reopen the terminal after installation, or restart the machine.

---

### Problem — `Cannot connect to the Docker daemon`

- **Windows / macOS**: open Docker Desktop and wait for the green status.
- **Linux**: `sudo systemctl start docker`.

---

### Problem — Port already allocated

Something else on the host already uses port 5000 / 8000 / 8501. Either stop that program, or change the host port in `docker-compose.yml`:

```yaml
ports:
  - "15000:5000"
```

---

### Problem — FastAPI cannot reach MLflow

Symptoms: a `ConnectionRefusedError` in FastAPI logs.

Checks:

- the env var must be `MLFLOW_TRACKING_URI=http://mlflow:5000` (service name, not `localhost`);
- the MLflow container is `healthy`: `docker compose ps`;
- both services are on the same network: `docker network inspect mlops-stack_mlops-net`.

---

### Problem — Code changes are not picked up

By default, the Python code is **copied** into the image at build time. After editing the source, rebuild:

```bash
docker compose up --build
```

For a smoother development loop, see **Appendix 03 — Hot Reload**.

---

### Problem — `no space left on device`

Docker images and volumes pile up over time. See **Appendix 02** for cleanup commands.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-17"></a>

## 17. Conclusion

You now have a complete MLOps stack running entirely in containers:

- a **single command** (`docker compose up`) starts MLflow, FastAPI, and Streamlit;
- the three services talk to each other through a private network;
- MLflow data is persisted on **named volumes** that survive `docker compose down`;
- the same `docker-compose.yml` reproduces the environment on any machine.

> [!IMPORTANT]
> Three rules to remember:
>
> 1. inside a container, listen on `0.0.0.0`, never on `127.0.0.1`;
> 2. between containers, call services by their **service name**, never by `localhost`;
> 3. anything that must survive a restart goes into a **volume**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-00"></a>

## 18. Appendix 00 — Useful Docker and Compose Commands

```bash
docker compose up              # start (foreground), build only if needed
docker compose up -d           # start in the background
docker compose up --build      # force rebuild before start
docker compose down            # stop and remove containers + network
docker compose down -v         # also remove the volumes (data loss!)
docker compose ps              # list services and their state
docker compose logs            # all logs, all services
docker compose logs -f mlflow  # follow logs of one service
docker compose exec fastapi bash   # open a shell inside a service
docker compose restart fastapi     # restart one service
docker compose build fastapi       # rebuild only one image
```

```bash
docker images                  # local images
docker ps                      # running containers
docker ps -a                   # all containers
docker volume ls               # all volumes
docker network ls              # all networks
docker stats                   # live CPU / RAM per container
```

> [!NOTE]
> `docker compose` (with a space) is the modern syntax. The old `docker-compose` (with a dash) still works on most installs but is being deprecated.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-01"></a>

## 19. Appendix 01 — `Dockerfile` vs `docker-compose.yml`

| File | Role | Scope |
| --- | --- | --- |
| `Dockerfile` | Describes how to **build one image** | A single service |
| `docker-compose.yml` | Describes how to **run several containers together** | The whole stack |

In short:

- a `Dockerfile` answers: *"How do I package this app?"*
- a `docker-compose.yml` answers: *"How do these apps run together?"*

> [!IMPORTANT]
> Each service in this practical work has its own `Dockerfile`. The `docker-compose.yml` does not replace them — it **uses** them through the `build:` directive.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-02"></a>

## 20. Appendix 02 — Cleaning Up Docker (Disk Space)

Over time, Docker keeps stopped containers, unused images, dangling volumes, and build cache. To reclaim disk space:

```bash
docker system df               # see disk usage by category
docker container prune         # remove stopped containers
docker image prune              # remove dangling images
docker image prune -a           # remove all unused images
docker volume prune             # remove unused volumes
docker builder prune            # remove build cache
docker system prune             # all of the above (containers/images/networks)
docker system prune -a --volumes  # nuclear option
```

> [!CAUTION]
> `docker system prune -a --volumes` deletes **everything** that is not currently in use, including all dangling volumes. Make sure you do not need that data anymore.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-03"></a>

## 21. Appendix 03 — Hot Reload for Development

For a faster development loop, mount the local source code into the container and run the server with auto-reload.

### 21.1 FastAPI with auto-reload

In `docker-compose.yml`, override the FastAPI service:

```yaml
  fastapi:
    build:
      context: ./fastapi
    volumes:
      - ./fastapi/app:/code/app
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

Now editing `fastapi/app/main.py` on the host immediately restarts uvicorn inside the container.

### 21.2 Streamlit with file watching

```yaml
  streamlit:
    build:
      context: ./streamlit
    volumes:
      - ./streamlit/app:/code/app
    command: ["streamlit", "run", "app/app.py",
              "--server.address=0.0.0.0",
              "--server.port=8501",
              "--server.runOnSave=true"]
```

> [!NOTE]
> Hot reload is great for development. **Do not** mount source code as a volume in production: production images must be self-contained.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-04"></a>

## 22. Appendix 04 — WSL 2 Backend on Windows

If Docker Desktop reports *"WSL 2 installation is incomplete"*:

1. Open **PowerShell as Administrator**:

   ```powershell
   wsl --install
   wsl --set-default-version 2
   wsl --update
   ```

2. Restart the computer.
3. Open **Docker Desktop → Settings → General** and check **"Use the WSL 2 based engine"**.
4. In **Settings → Resources → WSL Integration**, enable integration with your Ubuntu distribution.

> [!IMPORTANT]
> WSL 2 must be enabled in BIOS as well: the option is usually called **Virtualization Technology (VT-x / SVM)**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-05"></a>

## 23. Appendix 05 — Going to Production (PostgreSQL + S3)

For a real production deployment, replace SQLite and the local `mlruns` folder by:

- a real database (**PostgreSQL**) for the backend store;
- an object store (**S3**, **MinIO**, **GCS**, **Azure Blob**) for artifacts.

A production-leaning Compose snippet:

```yaml
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - pg-data:/var/lib/postgresql/data
    networks:
      - mlops-net

  mlflow:
    build:
      context: ./mlflow
    environment:
      MLFLOW_BACKEND_STORE_URI: "postgresql://mlflow:mlflow@postgres:5432/mlflow"
      MLFLOW_DEFAULT_ARTIFACT_ROOT: "s3://my-mlflow-artifacts/"
      AWS_ACCESS_KEY_ID: "${AWS_ACCESS_KEY_ID}"
      AWS_SECRET_ACCESS_KEY: "${AWS_SECRET_ACCESS_KEY}"
    depends_on:
      - postgres
    ports:
      - "5000:5000"
    networks:
      - mlops-net

volumes:
  pg-data:
```

The MLflow `Dockerfile` for production should also install `psycopg2-binary` and `boto3`:

```dockerfile
RUN pip install --no-cache-dir mlflow==2.16.2 psycopg2-binary boto3
```

> [!IMPORTANT]
> SQLite is fine for learning and small teams. Beyond a few concurrent users, switch to PostgreSQL or MySQL. SQLite locks the entire database on writes and quickly becomes a bottleneck.

> [!NOTE]
> Secrets such as `AWS_ACCESS_KEY_ID` should be provided through a `.env` file or a secret manager — never committed to Git.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Practical Work 2 — Installing Docker Desktop and Running MLflow + FastAPI + Streamlit with Docker Compose</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
