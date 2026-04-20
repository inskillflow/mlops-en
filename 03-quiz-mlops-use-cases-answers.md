<a id="top"></a>

# Quiz — MLOps — ANSWER KEY

## Theory Only — No Code

> **For the instructor only.**
> Each answer is hidden inside a `<details>` block. Click the arrow to reveal it.

---

## Table of Contents

| #  | Section                                                      |
| -- | ------------------------------------------------------------ |
| 1  | [Section A — Definitions and Purpose](#sa)                   |
| 2  | [Section B — Why MLOps Exists](#sb)                          |
| 3  | [Section C — DevOps vs MLOps](#sc)                           |
| 4  | [Section D — Lifecycle of an ML System](#sd)                 |
| 5  | [Section E — Components of MLOps](#se)                       |
| 6  | [Section F — Scenarios Without MLOps](#sf)                   |
| 7  | [Section G — Scenarios With MLOps](#sg)                      |
| 8  | [Section H — Monitoring, Drift, and Retraining](#sh)         |
| 9  | [Section I — Traceability, Governance, and Reliability](#si) |
| 10 | [Section J — True or False — Justify](#sj)                   |

---

<a id="sa"></a>

## Section A — Definitions and Purpose

---

**A1.** What does **MLOps** mean? Write the full term and explain it in your own words.

<details>
<summary>Answer</summary>

**MLOps** stands for **Machine Learning Operations**.

It is the discipline used to organize, automate, deploy, monitor, and maintain machine learning systems in real production environments. It extends machine learning work beyond experimentation and helps make models reliable over time.

</details>

---

**A2.** What is the main purpose of MLOps in a machine learning project?

<details>
<summary>Answer</summary>

The main purpose of MLOps is to manage the **full lifecycle** of machine learning systems in a structured and reliable way.

This includes training, validation, deployment, monitoring, versioning, retraining, and governance. Its goal is to make ML systems reproducible, traceable, maintainable, and usable in production.

</details>

---

**A3.** Explain why training a model is not the end of a machine learning project.

<details>
<summary>Answer</summary>

Training is only one stage. After a model is trained, the team still needs to verify that it is good enough, deploy it into a real environment, monitor its behavior, detect data changes, and retrain it when necessary.

A trained model has value only if it can continue to work correctly after deployment.

</details>

---

**A4.** What does it mean to say that MLOps helps turn machine learning from an experiment into a production system?

<details>
<summary>Answer</summary>

It means that MLOps takes a model that may work in a notebook or lab setting and makes it operational in the real world.

Instead of staying as a one-time experiment, the model becomes part of a managed system with deployment, monitoring, version control, and maintenance processes.

</details>

---

**A5.** Name at least four activities that MLOps helps manage after a model has been trained.

<details>
<summary>Answer</summary>

Possible correct answers include:

* model evaluation
* deployment
* monitoring
* version tracking
* drift detection
* retraining
* governance
* traceability
* performance review in production

Any four valid lifecycle activities are acceptable.

</details>

---

**A6.** Why is reproducibility important in MLOps?

<details>
<summary>Answer</summary>

Reproducibility is important because teams must be able to rebuild or explain a model using the same code, data, parameters, and process.

Without reproducibility, it becomes difficult to debug problems, compare versions fairly, trust results, or justify why a certain model was deployed.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sb"></a>

## Section B — Why MLOps Exists

---

**B1.** Why do machine learning systems create operational problems that traditional software systems do not always have?

<details>
<summary>Answer</summary>

Machine learning systems do not depend only on code. They also depend on data, features, training conditions, and learned model behavior.

Because of this, the system can change or degrade even when the code itself has not changed. This creates extra operational challenges such as model drift, retraining needs, and experiment tracking.

</details>

---

**B2.** Explain why a model can work well during testing but perform poorly in production.

<details>
<summary>Answer</summary>

A model may perform well during testing because the test data is similar to the training data and the experimental setup is controlled.

In production, the incoming data may be different, preprocessing may not match the training pipeline, or real-world conditions may be more complex. As a result, the model can perform worse after deployment.

</details>

---

**B3.** Why can it be dangerous if a team does not know which model version is currently deployed?

<details>
<summary>Answer</summary>

If the deployed model version is unknown, the team cannot clearly explain predictions, investigate issues, compare results, or roll back safely.

This creates serious traceability and reliability problems, especially when decisions depend on the model.

</details>

---

**B4.** What does it mean when people say that machine learning systems can “degrade silently”?

<details>
<summary>Answer</summary>

It means the model may continue running without crashes or technical errors, but the quality of its predictions becomes worse over time.

This degradation can go unnoticed unless monitoring is in place. The application may look healthy from a software perspective while the model is becoming less useful.

</details>

---

**B5.** Why is MLOps especially important when the incoming real-world data changes over time?

<details>
<summary>Answer</summary>

Because machine learning models are trained on past data. If real-world input patterns change, the model may no longer reflect reality.

MLOps helps detect those changes through monitoring and drift detection, and supports retraining when needed.

</details>

---

**B6.** Give one concrete example of a business risk that could happen without MLOps.

<details>
<summary>Answer</summary>

Example: a fraud detection model becomes outdated and starts missing fraudulent transactions because no one monitors drift or retrains it.

Other valid examples include poor recommendations in e-commerce, incorrect demand forecasts, or business decisions based on an untracked model version.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sc"></a>

## Section C — DevOps vs MLOps

---

**C1.** In your own words, what is the main difference between **DevOps** and **MLOps**?

<details>
<summary>Answer</summary>

**DevOps** focuses on building, testing, and deploying software systems.

**MLOps** includes those operational ideas but adds the management of data, training, experiments, model versions, monitoring of prediction quality, and retraining. The key idea is that MLOps manages machine learning systems, not only software code.

</details>

---

**C2.** DevOps mainly manages software systems. What additional elements must MLOps manage?

<details>
<summary>Answer</summary>

MLOps must manage:

* datasets
* feature engineering logic
* experiment tracking
* trained model artifacts
* model versions
* prediction quality in production
* data drift and model drift
* retraining workflows

These elements go beyond traditional software delivery.

</details>

---

**C3.** Why is DevOps alone often not enough for machine learning systems?

<details>
<summary>Answer</summary>

DevOps is strong for software delivery, but machine learning systems have extra problems related to data and learned behavior.

A model can degrade, drift, or need retraining even if the application is still running correctly. DevOps does not fully address those ML-specific concerns by itself.

</details>

---

**C4.** Explain the difference between a system whose behavior depends mainly on code and a system whose behavior depends on code, data, and trained models.

<details>
<summary>Answer</summary>

In a traditional software system, behavior is mainly determined by rules explicitly written in code.

In a machine learning system, behavior depends not only on code but also on the data used for training and the model learned from that data. This makes the system more sensitive to data changes and statistical variation.

</details>

---

**C5.** What is **Continuous Training**, and why is it much more relevant in MLOps than in DevOps?

<details>
<summary>Answer</summary>

**Continuous Training** means retraining models regularly or when conditions require it, such as new data arrival or drift detection.

It is much more relevant in MLOps because ML systems must adapt to changing data and conditions. Traditional DevOps usually focuses on code changes, not retraining learned models.

</details>

---

**C6.** A team says: “Our application is running, so everything is fine.”
Why can this statement be misleading in a machine learning context?

<details>
<summary>Answer</summary>

Because a machine learning system can be technically available while its predictions are becoming worse.

The server may be up, the API may respond correctly, and the logs may show no error, yet the model may still be inaccurate or outdated. Operational health and model quality are not the same thing.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sd"></a>

## Section D — Lifecycle of an ML System

---

**D1.** List the main stages of a typical machine learning lifecycle managed through MLOps.

<details>
<summary>Answer</summary>

A correct sequence usually includes:

* business problem definition
* data collection
* data preparation
* training
* evaluation
* deployment
* monitoring
* retraining or updating

Equivalent wording is acceptable if the lifecycle logic is clear.

</details>

---

**D2.** What is the role of data collection in the lifecycle of an ML system?

<details>
<summary>Answer</summary>

Data collection provides the raw information used to train and evaluate the model.

If the collected data is poor, incomplete, biased, or outdated, the model built from it will also be weak. Data collection is therefore a foundational stage.

</details>

---

**D3.** Why is data preparation an important step before training?

<details>
<summary>Answer</summary>

Because raw data is often noisy, inconsistent, incomplete, or not directly usable.

Data preparation helps clean, transform, validate, and organize the data so that the model can learn from reliable inputs.

</details>

---

**D4.** What is the purpose of model evaluation before deployment?

<details>
<summary>Answer</summary>

Model evaluation checks whether the trained model is good enough to be used.

It measures quality, robustness, and readiness, and helps prevent weak or unsafe models from being deployed.

</details>

---

**D5.** What is the difference between deployment and monitoring?

<details>
<summary>Answer</summary>

**Deployment** is the step where the model is made available in production.

**Monitoring** is the ongoing observation of that deployed model to see how it behaves over time, whether prediction quality is stable, and whether drift or other issues are appearing.

</details>

---

**D6.** Why is retraining considered part of the lifecycle and not an exceptional event?

<details>
<summary>Answer</summary>

Because real-world data and patterns often change over time. As a result, many deployed models eventually need updating.

Retraining is a normal operational need in machine learning, not just an occasional emergency action.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="se"></a>

## Section E — Components of MLOps

---

**E1.** What is the role of a **data pipeline** in MLOps?

<details>
<summary>Answer</summary>

A data pipeline moves, prepares, validates, and serves the data used by the ML system.

It helps ensure that data flows consistently from source to training or inference steps.

</details>

---

**E2.** What is the purpose of a **training pipeline**?

<details>
<summary>Answer</summary>

A training pipeline automates and standardizes the model training process.

It helps make training repeatable, controlled, and easier to compare across runs.

</details>

---

**E3.** What is **experiment tracking**, and why is it useful?

<details>
<summary>Answer</summary>

Experiment tracking means recording important information about training runs, such as parameters, datasets, metrics, and results.

It is useful because it allows teams to compare experiments, reproduce outcomes, and understand how a model was obtained.

</details>

---

**E4.** What is a **model registry**, and what problem does it solve?

<details>
<summary>Answer</summary>

A model registry is a controlled place where model versions are stored, identified, and tracked.

It solves the problem of version confusion by making it clear which model exists, which one is approved, and which one is deployed.

</details>

---

**E5.** Why is monitoring considered a core component of MLOps?

<details>
<summary>Answer</summary>

Because deployment alone is not enough. After deployment, the team must still observe model behavior, quality, and data conditions.

Monitoring helps detect degradation, drift, anomalies, and retraining needs.

</details>

---

**E6.** What is meant by **governance** in MLOps?

<details>
<summary>Answer</summary>

Governance refers to the rules, controls, approvals, documentation, and accountability around ML systems.

It ensures that models are reviewed, traceable, explainable, and managed responsibly.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sf"></a>

## Section F — Scenarios Without MLOps

---

**F1.** A data scientist trains a model in a notebook. It performs well there, but once deployed, it gives poor predictions because preprocessing was not applied correctly in production.
Explain what went wrong.

<details>
<summary>Answer</summary>

The training setup and production setup were not aligned.

The model depended on preprocessing steps that were not consistently packaged or reproduced in deployment. This is a classic notebook-to-production gap and a strong example of missing MLOps discipline.

</details>

---

**F2.** A company has trained three versions of a fraud detection model, but nobody knows which one is currently used in production.
What problem does this illustrate?

<details>
<summary>Answer</summary>

This illustrates a **model traceability and versioning problem**.

Without clear tracking or a model registry, the team cannot identify the deployed version, compare results reliably, or explain decisions.

</details>

---

**F3.** A recommendation model still runs every day, but its suggestions are becoming less useful and nobody notices.
Why is this an example of missing MLOps?

<details>
<summary>Answer</summary>

Because there is no monitoring of prediction quality or drift.

The system is operational from a software perspective, but the model is degrading silently. MLOps should detect this kind of performance decline.

</details>

---

**F4.** A team retrains a forecasting model, gets different results, but did not save the dataset version, training parameters, or metrics of the previous run.
Why does this create confusion?

<details>
<summary>Answer</summary>

Because the team cannot explain what changed or whether the new model is actually better.

Without tracked datasets, parameters, and metrics, comparison becomes unreliable and reproducibility is lost.

</details>

---

**F5.** During an internal audit, a team is asked who approved a production model and what tests were performed before deployment. They cannot answer clearly.
What important MLOps capability is missing?

<details>
<summary>Answer</summary>

The missing capability is **governance and traceability**.

The team lacks documented approval flow, testing history, and deployment accountability.

</details>

---

**F6.** In one or two sentences, explain the general consequence of running ML projects without MLOps.

<details>
<summary>Answer</summary>

Without MLOps, machine learning projects often remain manual, fragile, poorly tracked, and difficult to maintain.

They become harder to trust, explain, reproduce, monitor, and improve over time.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sg"></a>

## Section G — Scenarios With MLOps

---

**G1.** A team uses a controlled pipeline that stores preprocessing logic, model version, training metrics, and deployment status.
Explain how this improves reliability compared with the previous section.

<details>
<summary>Answer</summary>

It improves reliability because the full path from training to production is controlled and documented.

The team can reproduce the same behavior, identify exactly what was deployed, and reduce mistakes caused by manual handoffs or undocumented steps.

</details>

---

**G2.** A model registry shows that version 2.3 is currently deployed, which dataset it used, and when it was approved.
Why is this valuable for the team?

<details>
<summary>Answer</summary>

It gives the team clear traceability and accountability.

They can answer operational questions quickly, investigate issues more effectively, compare models properly, and justify why that model is in production.

</details>

---

**G3.** A monitoring system detects that the input data distribution is changing and alerts the team before performance becomes too poor.
Why is this a strong example of MLOps value?

<details>
<summary>Answer</summary>

Because MLOps is not only about deployment. It is also about protecting model quality after deployment.

Detecting drift early allows the team to act before the model becomes too inaccurate or harmful.

</details>

---

**G4.** A retraining pipeline compares the current model against a new candidate model before promotion to production.
Why is this better than manual retraining without rules?

<details>
<summary>Answer</summary>

Because the process becomes controlled, measurable, and defensible.

Instead of guessing, the team can compare metrics, apply standards, and promote only models that actually improve the system.

</details>

---

**G5.** A team can clearly explain when a model was trained, how it was evaluated, who approved it, and why it is in production.
What does this show about the maturity of the ML system?

<details>
<summary>Answer</summary>

It shows that the ML system has strong operational maturity.

The system is not only technically functional, but also traceable, governed, and responsibly managed.

</details>

---

**G6.** In your own words, summarize the difference between a scenario **without MLOps** and a scenario **with MLOps**.

<details>
<summary>Answer</summary>

Without MLOps, ML work is often manual, unclear, and difficult to trust.

With MLOps, the system becomes structured, versioned, monitored, reproducible, and easier to maintain over time.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sh"></a>

## Section H — Monitoring, Drift, and Retraining

---

**H1.** What is **data drift**?

<details>
<summary>Answer</summary>

Data drift is a change in the characteristics or distribution of incoming data over time.

It means the real-world input data is no longer similar to the data the model was originally trained on.

</details>

---

**H2.** Why can a model become less useful over time even if the code does not change?

<details>
<summary>Answer</summary>

Because the world, user behavior, business conditions, or incoming data may change.

Since the model learned patterns from earlier data, it may become less relevant when those patterns evolve.

</details>

---

**H3.** Why is model monitoring necessary after deployment?

<details>
<summary>Answer</summary>

Monitoring is necessary because deployment is not the final step.

The team must continue checking whether the model is still accurate, whether the data has drifted, and whether production behavior remains acceptable.

</details>

---

**H4.** What types of signals might indicate that a model should be retrained?

<details>
<summary>Answer</summary>

Possible valid signals include:

* declining prediction quality
* data drift
* model drift
* lower business KPIs
* new available data
* changing user behavior
* environment or domain changes

Any clear retraining signal is acceptable.

</details>

---

**H5.** Why is retraining not enough by itself if it is not tracked and evaluated properly?

<details>
<summary>Answer</summary>

Because retraining can produce a different model, but not necessarily a better one.

If retraining is not documented and evaluated, the team cannot know whether the update improved the system or introduced new problems.

</details>

---

**H6.** Explain the relationship between monitoring, drift detection, and retraining.

<details>
<summary>Answer</summary>

Monitoring observes the deployed model and data in real conditions.

Drift detection is one result of monitoring that identifies important data or behavior changes. Retraining is then the corrective action used when those changes make the current model less suitable.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="si"></a>

## Section I — Traceability, Governance, and Reliability

---

**I1.** What does **traceability** mean in MLOps?

<details>
<summary>Answer</summary>

Traceability means being able to connect a deployed model back to its code version, dataset version, training conditions, metrics, and approval history.

It allows the team to understand exactly where a model came from and how it reached production.

</details>

---

**I2.** Why is it important to connect a deployed model to its data version, code version, and evaluation results?

<details>
<summary>Answer</summary>

Because this information is necessary to explain model behavior, compare versions, debug problems, reproduce results, and justify deployment decisions.

Without those links, the system becomes much harder to trust and manage.

</details>

---

**I3.** Why is governance important when machine learning models are used in real organizational decisions?

<details>
<summary>Answer</summary>

Because decisions based on ML can affect users, business outcomes, or operations.

Governance ensures that models are reviewed, documented, approved, and controlled rather than used without accountability.

</details>

---

**I4.** Explain how MLOps helps increase trust in machine learning systems.

<details>
<summary>Answer</summary>

MLOps increases trust by making the system more reproducible, traceable, monitored, and governed.

Teams know what was trained, what was deployed, how it performs, and when it needs updating.

</details>

---

**I5.** What is the difference between a model that is merely “working” and a model that is “operationally reliable”?

<details>
<summary>Answer</summary>

A model that is merely “working” can produce predictions right now.

A model that is operationally reliable is tracked, monitored, explainable, versioned, and maintained so the organization can depend on it over time.

</details>

---

**I6.** In one paragraph, explain why MLOps is essential for maintaining machine learning systems over time.

<details>
<summary>Answer</summary>

MLOps is essential because machine learning systems do not remain stable automatically after deployment. Their quality depends on data, model versions, and changing real-world conditions. MLOps provides the structure needed to version models, monitor performance, detect drift, retrain when necessary, and maintain traceability and governance. Without it, machine learning systems become fragile and difficult to trust in production.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sj"></a>

## Section J — True or False — Justify

---

**J1.** If a machine learning model was accurate during training, it will remain accurate forever in production.

<details>
<summary>Answer</summary>

**FALSE**

Production conditions can change over time. Data drift, behavior changes, or new patterns can reduce model accuracy even if the original training performance was strong.

</details>

---

**J2.** DevOps and MLOps are exactly the same thing.

<details>
<summary>Answer</summary>

**FALSE**

MLOps builds on some DevOps ideas, but it also manages data, models, experiment tracking, drift, and retraining, which are specific to machine learning systems.

</details>

---

**J3.** In machine learning systems, the data can be as important as the code.

<details>
<summary>Answer</summary>

**TRUE**

The model learns behavior from data. Poor, outdated, or drifting data can seriously affect system quality even if the code is correct.

</details>

---

**J4.** A machine learning model can become less useful even when the application itself is still technically running.

<details>
<summary>Answer</summary>

**TRUE**

An API or application can remain available while the model’s prediction quality declines. Technical uptime does not guarantee model usefulness.

</details>

---

**J5.** Monitoring is optional because once a model is deployed, the main work is finished.

<details>
<summary>Answer</summary>

**FALSE**

Monitoring is essential after deployment because the model may drift, degrade, or face changing data. Deployment is not the end of ML operations.

</details>

---

**J6.** A model registry helps teams know which model version is in production.

<details>
<summary>Answer</summary>

**TRUE**

A model registry exists specifically to identify, store, and track model versions, including approval and deployment status.

</details>

---

**J7.** Retraining a model without recording the data and parameters used can create problems.

<details>
<summary>Answer</summary>

**TRUE**

Without tracking data and parameters, the team cannot reproduce the training run, compare versions reliably, or explain why results changed.

</details>

---

**J8.** MLOps helps improve reproducibility, traceability, and reliability.

<details>
<summary>Answer</summary>

**TRUE**

These are core goals of MLOps. It exists to make ML systems more structured, trustworthy, and maintainable in production.

</details>

---

**J9.** Governance in MLOps has nothing to do with approval, control, or documentation.

<details>
<summary>Answer</summary>

**FALSE**

Governance is directly related to approval, control, documentation, accountability, and responsible management of ML systems.

</details>

---

**J10.** Without MLOps, machine learning projects are more likely to stay manual, fragile, and difficult to maintain.

<details>
<summary>Answer</summary>

**TRUE**

Without structured operations, ML systems often suffer from poor traceability, weak monitoring, and inconsistent deployment or retraining practices.

</details>

---

**J11.** Continuous Training is more relevant to MLOps than to traditional DevOps.

<details>
<summary>Answer</summary>

**TRUE**

Continuous Training addresses the need to retrain models as data changes, which is a machine learning concern rather than a standard software delivery concern.

</details>

---

**J12.** MLOps exists mainly to replace machine learning.

<details>
<summary>Answer</summary>

**FALSE**

MLOps does not replace machine learning. It supports and operationalizes machine learning so that models can be used and maintained properly in real environments.

</details>

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Answer Key — MLOps Quiz</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
