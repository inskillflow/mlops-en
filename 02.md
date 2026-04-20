<a id="top"></a>

# Quiz — MLOps

## Theory Only — No Code

> **Instructions:**
> Answer every question in writing. No multiple choice — you write the answer yourself.
> This quiz is entirely theoretical. No code, no tools, no diagrams are required in your answers.
> Refer to nothing except your own knowledge.

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

```text
Your answer:
```

---

**A2.** What is the main purpose of MLOps in a machine learning project?

```text
Your answer:
```

---

**A3.** Explain why training a model is not the end of a machine learning project.

```text
Your answer:
```

---

**A4.** What does it mean to say that MLOps helps turn machine learning from an experiment into a production system?

```text
Your answer:
```

---

**A5.** Name at least four activities that MLOps helps manage after a model has been trained.

```text
Your answer:
```

---

**A6.** Why is reproducibility important in MLOps?

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sb"></a>

## Section B — Why MLOps Exists

---

**B1.** Why do machine learning systems create operational problems that traditional software systems do not always have?

```text
Your answer:
```

---

**B2.** Explain why a model can work well during testing but perform poorly in production.

```text
Your answer:
```

---

**B3.** Why can it be dangerous if a team does not know which model version is currently deployed?

```text
Your answer:
```

---

**B4.** What does it mean when people say that machine learning systems can “degrade silently”?

```text
Your answer:
```

---

**B5.** Why is MLOps especially important when the incoming real-world data changes over time?

```text
Your answer:
```

---

**B6.** Give one concrete example of a business risk that could happen without MLOps.

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sc"></a>

## Section C — DevOps vs MLOps

---

**C1.** In your own words, what is the main difference between **DevOps** and **MLOps**?

```text
Your answer:
```

---

**C2.** DevOps mainly manages software systems. What additional elements must MLOps manage?

```text
Your answer:
```

---

**C3.** Why is DevOps alone often not enough for machine learning systems?

```text
Your answer:
```

---

**C4.** Explain the difference between a system whose behavior depends mainly on code and a system whose behavior depends on code, data, and trained models.

```text
Your answer:
```

---

**C5.** What is **Continuous Training**, and why is it much more relevant in MLOps than in DevOps?

```text
Your answer:
```

---

**C6.** A team says: “Our application is running, so everything is fine.”
Why can this statement be misleading in a machine learning context?

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sd"></a>

## Section D — Lifecycle of an ML System

---

**D1.** List the main stages of a typical machine learning lifecycle managed through MLOps.

```text
Your answer:
```

---

**D2.** What is the role of data collection in the lifecycle of an ML system?

```text
Your answer:
```

---

**D3.** Why is data preparation an important step before training?

```text
Your answer:
```

---

**D4.** What is the purpose of model evaluation before deployment?

```text
Your answer:
```

---

**D5.** What is the difference between deployment and monitoring?

```text
Your answer:
```

---

**D6.** Why is retraining considered part of the lifecycle and not an exceptional event?

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="se"></a>

## Section E — Components of MLOps

---

**E1.** What is the role of a **data pipeline** in MLOps?

```text
Your answer:
```

---

**E2.** What is the purpose of a **training pipeline**?

```text
Your answer:
```

---

**E3.** What is **experiment tracking**, and why is it useful?

```text
Your answer:
```

---

**E4.** What is a **model registry**, and what problem does it solve?

```text
Your answer:
```

---

**E5.** Why is monitoring considered a core component of MLOps?

```text
Your answer:
```

---

**E6.** What is meant by **governance** in MLOps?

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sf"></a>

## Section F — Scenarios Without MLOps

---

**F1.** A data scientist trains a model in a notebook. It performs well there, but once deployed, it gives poor predictions because preprocessing was not applied correctly in production.
Explain what went wrong.

```text
Your answer:
```

---

**F2.** A company has trained three versions of a fraud detection model, but nobody knows which one is currently used in production.
What problem does this illustrate?

```text
Your answer:
```

---

**F3.** A recommendation model still runs every day, but its suggestions are becoming less useful and nobody notices.
Why is this an example of missing MLOps?

```text
Your answer:
```

---

**F4.** A team retrains a forecasting model, gets different results, but did not save the dataset version, training parameters, or metrics of the previous run.
Why does this create confusion?

```text
Your answer:
```

---

**F5.** During an internal audit, a team is asked who approved a production model and what tests were performed before deployment. They cannot answer clearly.
What important MLOps capability is missing?

```text
Your answer:
```

---

**F6.** In one or two sentences, explain the general consequence of running ML projects without MLOps.

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sg"></a>

## Section G — Scenarios With MLOps

---

**G1.** A team uses a controlled pipeline that stores preprocessing logic, model version, training metrics, and deployment status.
Explain how this improves reliability compared with the previous section.

```text
Your answer:
```

---

**G2.** A model registry shows that version 2.3 is currently deployed, which dataset it used, and when it was approved.
Why is this valuable for the team?

```text
Your answer:
```

---

**G3.** A monitoring system detects that the input data distribution is changing and alerts the team before performance becomes too poor.
Why is this a strong example of MLOps value?

```text
Your answer:
```

---

**G4.** A retraining pipeline compares the current model against a new candidate model before promotion to production.
Why is this better than manual retraining without rules?

```text
Your answer:
```

---

**G5.** A team can clearly explain when a model was trained, how it was evaluated, who approved it, and why it is in production.
What does this show about the maturity of the ML system?

```text
Your answer:
```

---

**G6.** In your own words, summarize the difference between a scenario **without MLOps** and a scenario **with MLOps**.

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sh"></a>

## Section H — Monitoring, Drift, and Retraining

---

**H1.** What is **data drift**?

```text
Your answer:
```

---

**H2.** Why can a model become less useful over time even if the code does not change?

```text
Your answer:
```

---

**H3.** Why is model monitoring necessary after deployment?

```text
Your answer:
```

---

**H4.** What types of signals might indicate that a model should be retrained?

```text
Your answer:
```

---

**H5.** Why is retraining not enough by itself if it is not tracked and evaluated properly?

```text
Your answer:
```

---

**H6.** Explain the relationship between monitoring, drift detection, and retraining.

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="si"></a>

## Section I — Traceability, Governance, and Reliability

---

**I1.** What does **traceability** mean in MLOps?

```text
Your answer:
```

---

**I2.** Why is it important to connect a deployed model to its data version, code version, and evaluation results?

```text
Your answer:
```

---

**I3.** Why is governance important when machine learning models are used in real organizational decisions?

```text
Your answer:
```

---

**I4.** Explain how MLOps helps increase trust in machine learning systems.

```text
Your answer:
```

---

**I5.** What is the difference between a model that is merely “working” and a model that is “operationally reliable”?

```text
Your answer:
```

---

**I6.** In one paragraph, explain why MLOps is essential for maintaining machine learning systems over time.

```text
Your answer:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<a id="sj"></a>

## Section J — True or False — Justify

**Write TRUE or FALSE, then write one sentence explaining why.**

---

**J1.** If a machine learning model was accurate during training, it will remain accurate forever in production.

```text
True or False:

Justification:
```

---

**J2.** DevOps and MLOps are exactly the same thing.

```text
True or False:

Justification:
```

---

**J3.** In machine learning systems, the data can be as important as the code.

```text
True or False:

Justification:
```

---

**J4.** A machine learning model can become less useful even when the application itself is still technically running.

```text
True or False:

Justification:
```

---

**J5.** Monitoring is optional because once a model is deployed, the main work is finished.

```text
True or False:

Justification:
```

---

**J6.** A model registry helps teams know which model version is in production.

```text
True or False:

Justification:
```

---

**J7.** Retraining a model without recording the data and parameters used can create problems.

```text
True or False:

Justification:
```

---

**J8.** MLOps helps improve reproducibility, traceability, and reliability.

```text
True or False:

Justification:
```

---

**J9.** Governance in MLOps has nothing to do with approval, control, or documentation.

```text
True or False:

Justification:
```

---

**J10.** Without MLOps, machine learning projects are more likely to stay manual, fragile, and difficult to maintain.

```text
True or False:

Justification:
```

---

**J11.** Continuous Training is more relevant to MLOps than to traditional DevOps.

```text
True or False:

Justification:
```

---

**J12.** MLOps exists mainly to replace machine learning.

```text
True or False:

Justification:
```

---

<p align="right"><a href="#top">↑ Back to top</a></p>

---

<p align="center">
  <strong>End of Quiz — MLOps</strong><br/>
  <a href="#top">↑ Back to the top</a>
</p>
