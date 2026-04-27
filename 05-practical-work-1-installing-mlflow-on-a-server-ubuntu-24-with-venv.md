<a id="top"></a>

# Practical Work — Installing MLflow on an a server Ubuntu Virtual Machine with a Virtual Environment

## Table of Contents

| # | Section |
|---|---|
| 1 | [Introduction](#section-1) |
| 2 | [Prerequisites](#section-2) |
| 3 | [Important Rule — Always Use a Virtual Environment](#section-3) |
| 4 | [Installation Steps](#section-4) |
| 5 | [Exercise 1 — Compare MLflow Commands](#section-5) |
| 6 | [Exercise 2 — Run a Hello World Experiment](#section-6) |
| 7 | [Accessing the MLflow Interface](#section-7) |
| 8 | [Basic MLflow Usage](#section-8) |
| 9 | [Troubleshooting](#section-9) |
| 10 | [Additional Resources](#section-10) |
| 11 | [Conclusion](#section-11) |
| 12 | [Appendix 00 — Install Python 3.9 and Create a Virtual Environment on Ubuntu 22.04](#appendix-00) |
| 13 | [Appendix 01 — The Experiment ID / Experiment Context Is Required](#appendix-01) |
| 14 | [Appendix 02 — Difference Between `mlflow server` and `mlflow ui`](#appendix-02) |
| 15 | [Appendix 03 — Server Error](#appendix-03) |
| 16 | [Appendix 04 — Answer to Exercise 1 — Comparison of the 4 Commands](#appendix-04) |
| 17 | [Appendix 05 — Using MLflow with a Database (Backend Store)](#appendix-05) |

---

<a id="section-1"></a>

## 1. Introduction

This guide explains how to install and use **MLflow** on an Ubuntu virtual machine.

The objective is to:

- install Python and the required tools;
- create and use a **virtual environment**;
- install MLflow inside that virtual environment;
- launch an MLflow tracking server;
- run a simple Python script that logs one run;
- view the result in the MLflow web interface.

> [!NOTE]
> This practical work is designed around the principle that **all Python-related work must be done inside a virtual environment**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Prerequisites

You need:

- an Ubuntu virtual machine;
- Internet access;
- `sudo` privileges.

> [!IMPORTANT]
> Recommended environments:
>
> - **Ubuntu 24.04** with **Python 3.12**
> - **Ubuntu 22.04** with **Python 3.9**
>
> If you want to run the project on **Ubuntu 22.04**, see **Appendix 00**.

> [!NOTE]
> This document focuses on Ubuntu. The commands and workflow are not meant for Windows.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Important Rule — Always Use a Virtual Environment

> [!IMPORTANT]
> In this practical work, you must always work inside a **virtual environment**.
>
> Do **not** install MLflow globally on the machine.
>
> Every `pip install` in this work must be executed **after activating the virtual environment**.

Why?

- it keeps the operating system clean;
- it avoids conflicts between projects;
- it makes the setup reproducible;
- it follows proper Python and MLOps practices.

> [!WARNING]
> If you install Python packages globally by mistake, you may create conflicts with other projects or with the system Python environment.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-4"></a>

## 4. Installation Steps

### 4.1 Update the system

```bash
sudo apt update
````

> [!WARNING]
> Do **not** run the following command unless explicitly requested:
>
> ```bash
> sudo apt upgrade -y
> ```

---

### 4.2 Install Python, pip, and create the virtual environment

MLflow requires Python. We will install Python 3, pip, and the package needed to create a virtual environment.

### Ubuntu 24.04

```bash
sudo apt install python3 python3-pip -y
python3 --version
sudo apt install python3.12-venv -y
python3 -m venv myenv
source myenv/bin/activate
python --version
```

> [!NOTE]
> After activation, the command `python` points to the Python interpreter inside the virtual environment.

> [!CAUTION]
> If needed, reinstall the virtual environment package:
>
> ```bash
> sudo apt install python3.12-venv -y
> ```

To leave the virtual environment:

```bash
deactivate
```

> [!IMPORTANT]
> From this point onward, all `pip install` commands must be executed only after running:
>
> ```bash
> source myenv/bin/activate
> ```

---

### 4.3 Install MLflow

> [!IMPORTANT]
> Make sure the virtual environment is activated before installing MLflow.

```bash
pip install mlflow
```

> [!NOTE]
> MLflow is installed inside the virtual environment, not globally.

---

### 4.4 Install additional dependencies (optional)

Depending on your use case, you may need additional libraries. For example, for scikit-learn integration:

```bash
pip install scikit-learn
```

> [!NOTE]
> Optional libraries should also be installed inside the same virtual environment.

---

### 4.5 Configure the MLflow tracking workspace

Create a directory to store your MLflow experiments:

```bash
mkdir ~/mlflow-experiments
cd ~/mlflow-experiments
```

Create the database folder and launch the MLflow tracking server:

```bash
mkdir database
mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5000
```

> [!IMPORTANT]
> This terminal becomes **Terminal 1** and must stay open.

> [!WARNING]
> Before launching `mlflow server`, make sure the virtual environment is still activated in this terminal.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-5"></a>

## 5. Exercise 1 — Compare MLflow Commands

Compare the following commands:

```bash
mlflow ui
```

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

```bash
mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5000
```

```bash
mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root ~/mlflow-experiments --host 0.0.0.0 --port 5000
```

### Difference between the commands

| Command                                                                                                                                  | Role                                                             | Backend (metadata)                  | Artifacts (files, models, etc.)                                           | Host / Access                              | Typical use                                   |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------ | --------------------------------------------- |
| `mlflow ui`                                                                                                                              | Quick local interface without a fully configured tracking server | default local storage in `./mlruns` | `./mlruns` in the current folder                                          | `127.0.0.1` (local only)                   | quick exploration on a personal machine       |
| `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000`                    | Simple MLflow server, everything local to the project            | `mlflow.db` in the current folder   | `./mlruns` in the current folder                                          | `127.0.0.1` (local only)                   | solo development, small project               |
| `mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5000`          | MLflow server accessible on the network                          | `mlflow.db` in `./database/`        | local `mlruns` folder via URI notation                                    | `0.0.0.0` (accessible from other machines) | small shared server, LAN usage                |
| `mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root ~/mlflow-experiments --host 0.0.0.0 --port 5000` | A slightly more production-oriented MLflow server                | `mlflow.db` in `./database/`        | global folder `~/mlflow-experiments` independent from the current project | `0.0.0.0` (network)                        | central server for multiple users or projects |

> [!NOTE]
> The answer to this exercise is provided in **Appendix 04**.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-6"></a>

## 6. Exercise 2 — Run a Hello World Experiment

### Step 1

> **Terminal 1**

Make sure the virtual environment is activated, then run:

```bash
mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root ~/mlflow-experiments --host 0.0.0.0 --port 5000
```

> [!WARNING]
> This command may not work in every environment exactly as written. If needed, use the alternatives shown later in the document.

---

### Step 2

> **Terminal 2**

Activate the same virtual environment, then create and run the script:

```bash
source ~/myenv/bin/activate
cd ~/mlflow-experiments
nano hello-world.py
python3 hello-world.py
```

Paste the following code into `hello-world.py`:

```python
import mlflow

# Optionally set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment name
mlflow.set_experiment("Default")

# Start a new MLflow run
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    mlflow.set_tag("tag1", "example")
```

> [!IMPORTANT]
> Terminal 2 must use the **same virtual environment** as Terminal 1.

---

### Step 3 — Optional SQLite database inspection

#### Go to the correct folder

Since the command uses:

```bash
mlflow server --backend-store-uri sqlite:///database/mlflow.db ...
```

the database is inside the `database/` folder:

```bash
cd ~/mlflow-experiments/database
ls
```

You should see:

```bash
mlflow.db
```

#### Install the SQLite tool

```bash
sudo apt update
sudo apt install sqlite3 -y
```

#### Open the database

```bash
sqlite3 mlflow.db
```

You will enter the `sqlite>` prompt.

#### Useful SQLite commands

```sql
.tables;
.schema experiments;
SELECT * FROM experiments;
SELECT * FROM runs LIMIT 5;
.quit
```

> [!CAUTION]
> Do not modify the database manually. Do not use `DELETE` or `UPDATE`. Use SQLite only to inspect the data.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-7"></a>

## 7. Accessing the MLflow Interface

By default, the MLflow server runs on port `5000`.

You can access it in your browser at:

```text
http://<YOUR-VM-IP>:5000
```

Replace `<YOUR-VM-IP>` with the IP address of your Ubuntu VM.

If you are working locally on the VM itself, you may also use:

```text
http://127.0.0.1:5000
```

> [!NOTE]
> If the server was started with `--host 127.0.0.1`, it will only be accessible locally.

> [!IMPORTANT]
> If the server was started with `--host 0.0.0.0`, it can be accessed from other machines on the network, depending on firewall and VM network configuration.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-8"></a>

## 8. Basic MLflow Usage

### Logging a first experiment

Open another terminal if needed, activate the virtual environment, and run the following commands:

```bash
apt-get update -y && apt-get install -y gcc && apt-get install -y git
nano hello-world.py
cd mlflow-experiments/
ls
cd database/
ls
cd ..
ls
nano hello-world.py
python hello-world.py
python3 hello-world.py
ls -la
cd ..
ls -la
source myenv/bin/activate
ls
cd mlflow-experiments/
ls
python3 hello-world.py
history
```

> [!WARNING]
> Do not run `python hello-world.py` before confirming that the virtual environment is activated and that `python` points to the interpreter inside `myenv`.

### PATH troubleshooting

If needed:

```bash
PATH=$PATH:/home/root/.local/bin
```

### Python file

```bash
nano hello-world.py
```

```python
import mlflow

# Optionally set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment name
mlflow.set_experiment("Default")

# Start a new MLflow run
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    mlflow.set_tag("tag1", "example")
```

Run it:

```bash
python3 hello-world.py
```

### If this command does not work

```bash
mlflow server --backend-store-uri sqlite:///database/mlflow.db  --default-artifact-root ~/mlflow-experiments --host 0.0.0.0 --port 5000
```

Use one of the following commands instead:

```bash
mlflow ui
```

or

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Then go to:

```text
http://127.0.0.1:5000
```

### Experiment tracking

You can view the logged experiments and metrics in the MLflow interface.

Use:

```bash
mlflow ui
```

Then go to:

```text
http://127.0.0.1:5000
```

> [!NOTE]
> `mlflow ui` is useful for quick local experimentation, while `mlflow server` is more suitable for configured tracking setups.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-9"></a>

## 9. Troubleshooting

### Problem — `mlflow` command not found

Make sure the virtual environment is activated:

```bash
source myenv/bin/activate
```

Then verify MLflow:

```bash
mlflow --version
```

> [!WARNING]
> If the virtual environment is not active, the shell may not find the `mlflow` executable.

---

### Problem — Unable to connect to the server at `127.0.0.1:5000`

Make sure the server is running before executing the Python script.

A working local server command is:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

> [!NOTE]
> This is a local-only configuration because the host is set to `127.0.0.1`.

---

### Problem — Port 5000 is already in use

Check:

```bash
lsof -i :5000
```

If needed, use another port such as `5001`, and update the Python code:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5001")
```

---

### Problem — Browser connection issue

Sometimes the browser session may simply need to be refreshed or restarted.

> [!NOTE]
> If you use a remote VM, also verify VM networking rules and firewall settings.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-10"></a>

## 10. Additional Resources

* Official MLflow documentation
* MLflow tutorials and examples

> [!NOTE]
> Use the official documentation to explore more advanced features such as model registry, artifact logging, and remote backends.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-11"></a>

## 11. Conclusion

You have now installed and configured MLflow on an Ubuntu virtual machine while following the important rule of using a **virtual environment**.

You can now start tracking and managing machine learning experiments with MLflow.

> [!IMPORTANT]
> The most important operational rule in this work is simple:
>
> * create the virtual environment;
> * activate it;
> * install MLflow inside it;
> * run all Python and MLflow commands from that environment.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-00"></a>

## 12. Appendix 00 — Install Python 3.9 and Create a Virtual Environment on Ubuntu 22.04

```bash
rm -rf myenv
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev -y
python3.9 --version
python3.9 -m venv myenv
source myenv/bin/activate
```

> [!IMPORTANT]
> On Ubuntu 22.04, this appendix is useful if your environment requires Python 3.9 instead of the default Python version.

> [!NOTE]
> After activation, you should still install MLflow with:
>
> ```bash
> pip install mlflow
> ```

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-01"></a>

## 13. Appendix 01 — The Experiment ID / Experiment Context Is Required

* MLflow must create or use an experiment before logging a run.
* It is necessary to define an experiment context, otherwise MLflow may attempt to use an existing experiment ID.

```python
import mlflow

# Create or select an experiment named "Default"
mlflow.set_experiment("Default")

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    mlflow.set_tag("tag1", "example")
```

### Explanation

1. `mlflow.set_experiment("Default")` ensures that MLflow creates the experiment if it does not exist yet.
2. Optionally, if you already know the experiment ID, you may specify it explicitly:

```python
with mlflow.start_run(experiment_id=1):
    mlflow.log_param("param1", 5)
```

> [!IMPORTANT]
> Setting the experiment explicitly is a good habit because it makes the logging context clear and reproducible.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-02"></a>

## 14. Appendix 02 — Difference Between `mlflow server` and `mlflow ui`

The main difference between `mlflow server` and `mlflow ui` is the level of configuration and deployment.

### `mlflow server`

This command starts a full MLflow tracking server. It can be used in more robust environments and allows configuration of:

* the backend store;
* the artifact store;
* the network host and port.

Example:

```bash
mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5000
```

Typical use:

* collaborative environment;
* configurable tracking infrastructure;
* centralized tracking.

### `mlflow ui`

This command starts a local and simplified MLflow interface.

Characteristics:

* local usage only;
* fewer configuration options;
* typically uses the default local `mlruns` folder.

Typical use:

* quick local testing;
* fast exploration without a full tracking server configuration.

### Summary

* **`mlflow server`** = robust, configurable, multi-user capable
* **`mlflow ui`** = simple, local, quick visualization

> [!NOTE]
> `mlflow ui` is often enough for a quick demonstration, but `mlflow server` is more appropriate for structured work.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-03"></a>

## 15. Appendix 03 — Server Error

### Example code

```python
import mlflow

# Optionally set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the experiment name
mlflow.set_experiment("Default")

# Start a new MLflow run
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    mlflow.set_tag("tag1", "example")
```

Make sure that the MLflow server is started before running this code.

If you see an error such as:

> Unable to connect to the server at `127.0.0.1:5000`

this means the MLflow server is not currently running on that address.

### Checks

1. Verify that the MLflow server is started:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

2. Check whether port `5000` is already used:

```bash
lsof -i :5000
```

3. If needed, use a different port and update the Python script accordingly:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5001")
```

4. Check local firewall restrictions if applicable.
5. Restart the browser if needed.

> [!WARNING]
> If the server is not running, the Python script cannot log anything to MLflow.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-04"></a>

## 16. Appendix 04 — Answer to Exercise 1 — Comparison of the 4 Commands

### 1. `mlflow ui`

* **Description**: starts a simple local MLflow user interface.
* **Characteristics**:

  * no fully configured tracking server;
  * uses `mlruns/` by default in the current folder;
  * no explicit backend store;
  * suitable for quick exploration.
* **Typical use**:

  * local demo;
  * quick test;
  * rapid exploration.

---

### 2. `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000`

* **Description**: starts a complete MLflow tracking server.
* **Characteristics**:

  * backend store: SQLite file `mlflow.db` in the current folder;
  * artifact root: local folder `./mlruns`;
  * host: local machine only;
  * port: 5000.
* **Typical use**:

  * local project;
  * persistent metadata through SQLite;
  * not accessible remotely.

---

### 3. `mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5000`

* **Description**: starts a network-accessible MLflow server.
* **Characteristics**:

  * backend store: SQLite file in `database/mlflow.db`;
  * artifact root: `file:mlruns`;
  * host: `0.0.0.0`, accessible from all network interfaces;
  * port: 5000.
* **Typical use**:

  * shared machine;
  * LAN environment;
  * container-compatible setup.

---

### 4. `mlflow server --backend-store-uri sqlite:///database/mlflow.db --default-artifact-root ~/mlflow-experiments --host 0.0.0.0 --port 5000`

* **Description**: similar to command 3, but with a more customized artifact storage path.
* **Characteristics**:

  * backend store: SQLite file in `database/mlflow.db`;
  * artifact root: `~/mlflow-experiments`;
  * host: `0.0.0.0`;
  * port: 5000.
* **Typical use**:

  * shared development environment;
  * more stable artifact storage folder;
  * multiple users or projects.

### Synthetic comparison

| Command     | Backend Store                  | Artifact Root          | Accessibility     | Use case                        |
| ----------- | ------------------------------ | ---------------------- | ----------------- | ------------------------------- |
| `mlflow ui` | default local storage          | default `./mlruns`     | local only        | quick exploration               |
| Command 2   | `sqlite:///mlflow.db`          | `./mlruns`             | local `127.0.0.1` | local solo project              |
| Command 3   | `sqlite:///database/mlflow.db` | `file:mlruns`          | network `0.0.0.0` | collaboration, containers       |
| Command 4   | `sqlite:///database/mlflow.db` | `~/mlflow-experiments` | network `0.0.0.0` | collaboration, dedicated folder |

### Final comparison

```text
+---------+-----------------------------------------+---------------------------+------------------+-----------------------------+
| Cmd N°  | Backend Store URI                       | Artifact Root             | Host / Access    | Recommended Usage           |
+---------+-----------------------------------------+---------------------------+------------------+-----------------------------+
| 1       | (default, implicit)                     | ./mlruns (default)        | 127.0.0.1        | quick local usage           |
|         |                                         |                           | local only       | simple interface            |
+---------+-----------------------------------------+---------------------------+------------------+-----------------------------+
| 2       | sqlite:///mlflow.db                     | ./mlruns                  | 127.0.0.1        | local project with tracking |
|         | file in current folder                  |                           | local only       | persistence with SQLite     |
+---------+-----------------------------------------+---------------------------+------------------+-----------------------------+
| 3       | sqlite:///database/mlflow.db            | file:mlruns               | 0.0.0.0          | shared local server         |
|         | file in database folder                 | explicit local URI        | network open     | container-compatible        |
+---------+-----------------------------------------+---------------------------+------------------+-----------------------------+
| 4       | sqlite:///database/mlflow.db            | ~/mlflow-experiments      | 0.0.0.0          | shared environment          |
|         | same backend as command 3               | user workspace            | network open     | centralized artifacts       |
+---------+-----------------------------------------+---------------------------+------------------+-----------------------------+
```

> [!NOTE]
> In a learning environment, understanding the difference between these four commands is more important than memorizing them mechanically.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="appendix-05"></a>

## 17. Appendix 05 — Using MLflow with a Database (Backend Store)

### 17.1 Why does MLflow need a database?

MLflow manages the lifecycle of machine learning experiments. To ensure:

* traceability;
* persistence;
* consultation of results;

it needs a structured storage system for metadata such as:

* experiments;
* runs;
* hyperparameters;
* metrics;
* artifact paths;
* status information;
* durations;
* users.

By default, `mlflow ui` uses local file-based storage. However, for real collaborative or reproducible use cases, it is recommended to use `mlflow server` with a dedicated backend store.

---

### 17.2 Compatible database options

| Type                | Example URI                                | Typical use       | Advantages                           | Limitations                           |
| ------------------- | ------------------------------------------ | ----------------- | ------------------------------------ | ------------------------------------- |
| SQLite (local file) | `sqlite:///mlflow.db`                      | local development | simple, portable                     | not suitable for concurrent heavy use |
| MySQL               | `mysql+pymysql://user:pass@host/db_name`   | collaborative use | scalable, robust                     | more configuration required           |
| PostgreSQL          | `postgresql://user:pass@host:port/db_name` | production        | secure, robust                       | requires a server installation        |
| Databricks          | platform-specific configuration            | enterprise cloud  | integrated with Databricks ecosystem | restricted to that ecosystem          |

---

### 17.3 Example with SQLite

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Explanation:

* `--backend-store-uri` defines the location of the database file;
* `--default-artifact-root` defines where artifacts are stored;
* `--host` defines access scope;
* `--port` defines the listening port.

This configuration persists important metadata even after a restart.

---

### 17.4 Multi-user environment

In a collaborative environment, it is better to use:

* `--host 0.0.0.0` to allow remote access;
* a remote database such as PostgreSQL or MySQL;
* an artifact store accessible by all users, such as NFS or object storage.

Example with PostgreSQL:

```bash
mlflow server \
  --backend-store-uri postgresql://user:pass@host:5432/mlflow_db \
  --default-artifact-root s3://mlflow-artifacts/ \
  --host 0.0.0.0 \
  --port 5000
```

---

### 17.5 Summary

| Command                            | Database required | Persistent       | Network accessible | Recommended for production       |
| ---------------------------------- | ----------------- | ---------------- | ------------------ | -------------------------------- |
| `mlflow ui`                        | No                | Yes, local files | No                 | No                               |
| `mlflow server` + SQLite           | Yes               | Yes              | Optional           | No, mainly local or small setups |
| `mlflow server` + PostgreSQL/MySQL | Yes               | Yes              | Yes                | Yes                              |

> [!IMPORTANT]
> A backend database becomes much more important when experiments must be shared, preserved, and queried over time.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Practical Work — Installing MLflow on Ubuntu with a Virtual Environment</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
