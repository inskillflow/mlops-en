# MLOps Stack — Command Cheat Sheet

> All commands must be run from the `chap06-mlops-stack/` folder (the one that contains `docker-compose.yml`).

---

## URLs after `docker compose up -d`

### From your host browser (Windows / macOS / Linux desktop)

| Service   | URL                                |
| --------- | ---------------------------------- |
| Streamlit | http://localhost:8501              |
| FastAPI   | http://localhost:8000              |
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| FastAPI health         | http://localhost:8000/health |
| MLflow UI | http://localhost:5000              |

> `localhost` and `127.0.0.1` are equivalent here.

### From another machine on the LAN

Replace `<HOST-IP>` with your machine's IP (find it with `ipconfig` on Windows or `ip a` on Linux):

| Service   | URL                                |
| --------- | ---------------------------------- |
| Streamlit | http://&lt;HOST-IP&gt;:8501              |
| FastAPI   | http://&lt;HOST-IP&gt;:8000/docs         |
| MLflow UI | http://&lt;HOST-IP&gt;:5000              |

### From inside a container (service-to-service)

Use the **service name**, never `localhost`:

| Caller    | Target    | URL inside the network         |
| --------- | --------- | ------------------------------ |
| Streamlit | FastAPI   | http://fastapi:8000            |
| FastAPI   | MLflow    | http://mlflow:5000             |
| (any)     | Streamlit | http://streamlit:8501          |

Test it from inside a container:

```bash
docker compose exec streamlit sh -c "wget -qO- http://fastapi:8000/health"
docker compose exec fastapi   sh -c "wget -qO- http://mlflow:5000/"
```

### Find the internal IP of each container

```bash
docker inspect -f "{{.Name}} -> {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" mlflow fastapi streamlit
docker network inspect chap06-mlops-stack_mlops-net
```

> [!NOTE]
> Internal container IPs (typically `172.x.x.x`) **change** on each `docker compose up`. Always prefer the **service name** over the IP.

### See published ports at a glance

```bash
docker compose ps
docker compose port mlflow 5000
docker compose port fastapi 8000
docker compose port streamlit 8501
```

---

## Start

```bash
docker compose up --build          # build + start (foreground)
docker compose up -d --build       # build + start (background, detached)
docker compose up -d               # start without rebuild
```

## Stop

```bash
# foreground : Ctrl+C, then:
docker compose down                # stop + remove containers + network (volumes kept)
docker compose down -v             # also delete the volumes (MLflow data lost!)
docker compose down --rmi all      # also delete the built images
docker compose down -v --rmi all   # nuclear : containers + network + volumes + images
```

## Status & logs

```bash
docker compose ps                  # services state
docker compose logs                # all logs
docker compose logs -f             # follow all logs
docker compose logs -f mlflow      # follow logs of one service
docker compose logs -f fastapi
docker compose logs -f streamlit
```

## Enter a running container (shell inside)

```bash
docker compose exec mlflow bash
docker compose exec fastapi bash
docker compose exec streamlit bash
# leave with: exit
```

If `bash` is not present in the slim image:

```bash
docker compose exec fastapi sh
```

Run a one-off command without entering:

```bash
docker compose exec fastapi python -c "import mlflow; print(mlflow.__version__)"
docker compose exec mlflow ls /mlflow/database
docker compose exec mlflow ls /mlflow/mlruns
```

## Restart / rebuild one service

```bash
docker compose restart fastapi
docker compose build fastapi
docker compose up -d --build fastapi
```

## Volumes (MLflow persistence)

```bash
docker volume ls
docker volume inspect chap06-mlops-stack_mlflow-db
docker volume inspect chap06-mlops-stack_mlflow-artifacts
docker volume rm chap06-mlops-stack_mlflow-db chap06-mlops-stack_mlflow-artifacts
```

## Images

```bash
docker images                                  # list local images
docker rmi mlops/fastapi:latest                # remove one image
docker rmi mlops/mlflow:latest mlops/fastapi:latest mlops/streamlit:latest
docker image prune                             # remove dangling images
docker image prune -a                          # remove all unused images
```

## Containers

```bash
docker ps                          # running containers
docker ps -a                       # all containers (incl. stopped)
docker stop mlflow fastapi streamlit
docker rm   mlflow fastapi streamlit
docker container prune             # remove all stopped containers
```

## Disk usage & cleanup

```bash
docker system df                   # disk usage by category
docker system prune                # containers + networks + dangling images
docker system prune -a             # also unused images
docker system prune -a --volumes   # nuclear : also unused volumes
docker builder prune               # build cache only
```

## Full reset (start from scratch)

```bash
docker compose down -v --rmi all
docker system prune -a --volumes -f
docker compose up --build
```

---

## Quick health checks

```bash
curl http://localhost:5000/                       # MLflow
curl http://localhost:8000/health                 # FastAPI
curl -X POST http://localhost:8000/log-run -H "Content-Type: application/json" -d "{\"features\":[1,2,3]}"
```
