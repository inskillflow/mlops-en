<a id="top"></a>

# Theoretical Notes — Introduction to MLflow and Its Role in MLOps

## Table of Contents

| # | Section |
|---|---|
| 1 | [Part 1 — What Is MLflow?](#section-1) |
| 2 | [Part 2 — MLflow and MLOps](#section-2) |
| 3 | [Part 3 — Summary of the Main Components of MLflow](#section-3) |

---

<a id="section-1"></a>

## 1. Part 1 — What Is MLflow?

**MLflow** is an **open-source platform** designed to **manage the lifecycle of machine learning models**.  
It is widely used to track experiments, reproduce results, deploy models, and organize end-to-end machine learning workflows.

> [!IMPORTANT]
> MLflow is not just a logging tool. It is a platform that helps structure and manage the full machine learning workflow.

### What is MLflow used for?

MLflow is mainly used for the following tasks:

### 1.1 Experiment Tracking (`MLflow Tracking`)

- It allows users to record **parameters**, **metrics**, **artifacts** such as models or images, and even the **source code** used in each experiment.
- It is useful for **comparing the performance** of several model runs.

> [!NOTE]
> This component is particularly useful when the same model is trained many times with different hyperparameters.

### 1.2 Model Management (`MLflow Models`)

- It supports the registration of machine learning models in several formats such as **TensorFlow**, **PyTorch**, and **Scikit-learn**.
- It provides an API to **load**, **evaluate**, and **deploy** models.

> [!IMPORTANT]
> This makes MLflow very practical in environments where models must be moved from experimentation to deployment.

### 1.3 Model Registry (`MLflow Model Registry`)

- It is used to **manage model versions**, add comments, attach metadata, and mark models with stages such as **Staging** or **Production**.
- It is very useful for **team collaboration** and controlled validation before deployment.

> [!NOTE]
> The registry helps teams avoid confusion when multiple versions of the same model exist.

### 1.4 Easy Deployment (`MLflow Projects` and `MLflow Serving`)

- MLflow allows fast deployment with commands such as `mlflow serve` in order to expose a model as a **REST API**.
- It can also integrate with platforms such as **Docker**, **Kubernetes**, and **SageMaker**.

> [!IMPORTANT]
> This makes MLflow useful not only for experimentation but also for operational deployment.

### Concrete Use Cases

- Train a model multiple times with different hyperparameters and **compare the results automatically**.
- Track the history of tested models and reproduce the best-performing versions.
- Deploy a model directly from a Jupyter notebook or a structured workflow.
- Work as a team and follow the evolution of models over time.

> [!NOTE]
> In practice, MLflow becomes especially valuable when machine learning work moves beyond one notebook and one person.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-2"></a>

## 2. Part 2 — MLflow and MLOps

**MLflow** is a **central tool in MLOps**.

### Difference Between MLOps and MLflow

**MLOps** means **Machine Learning Operations**.  
It refers to the practices, methods, and tools used to **industrialize machine learning**, in the same way that DevOps industrializes software development.

**MLflow**, on the other hand, is a **software component** inside this broader ecosystem.  
Its role is to **manage the lifecycle of machine learning models**.

> [!IMPORTANT]
> MLOps is the broader discipline.  
> MLflow is one concrete tool that helps implement part of that discipline.

### Direct Relationship Between MLOps and MLflow

| **MLOps** | **MLflow helps with...** |
|---|---|
| Experiment tracking | Logging runs, parameters, and metrics |
| Model versioning | Managing versions with the Model Registry |
| Reproducibility | Saving code, configuration, and experiment context |
| Model deployment | Serving a model easily through REST APIs and deployment tools |
| Collaboration | Sharing, reviewing, and organizing models across teams |

> [!NOTE]
> MLflow does not replace all of MLOps. It supports several important parts of it.

### Conclusion

> [!IMPORTANT]
> MLflow is an **MLOps tool** that helps with production readiness, experiment tracking, reproducibility, and model lifecycle management.

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="section-3"></a>

## 3. Part 3 — Summary of the Main Components of MLflow

**MLflow** is an open-source platform used to **track experiments**, **manage machine learning models**, **record model versions**, and **deploy models easily**.  
It helps ensure **reproducibility**, **traceability**, and **fast deployment** of machine learning models.

> [!IMPORTANT]
> The strength of MLflow comes from the fact that it combines tracking, organization, versioning, and deployment in a single platform.

### The 4 Main Components of MLflow

1. **Tracking**  
   Records parameters, metrics, and artifacts.

2. **Projects**  
   Standardizes machine learning workflows and execution environments.

3. **Models**  
   Stores, loads, and serves machine learning models.

4. **Model Registry**  
   Manages model versions and lifecycle stages such as **staging** and **production**.

> [!NOTE]
> These four components cover a large part of the operational needs encountered in machine learning projects.

### Final Summary

| Component | Main Role |
|---|---|
| **Tracking** | Record experiments, parameters, metrics, and artifacts |
| **Projects** | Standardize the execution of ML workflows |
| **Models** | Package and serve models |
| **Model Registry** | Manage model versions and deployment stages |

> [!IMPORTANT]
> If you remember only one idea, remember this:
>
> **MLflow helps structure the full lifecycle of machine learning models, from experimentation to deployment.**

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Theoretical Notes — Introduction to MLflow and Its Role in MLOps</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
