---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.1rem;
  }
  h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 0.2em; }
  h2 { color: #16213e; }
  h3 { color: #e94560; }
  code { background: #f4f4f4; padding: 0.1em 0.4em; border-radius: 4px; }
  table { font-size: 0.85rem; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
---

# ZAP MLOps Foundations
## Modules 1–3 Summary

**EDA · Model Training · Evaluation & Validation**

---

# The 6-Week Journey

| Week | Module | Focus |
|------|--------|-------|
| 1 | **EDA** | Explore data, quality checks, train/val/test splits |
| 2 | **Model Training** | Data prep, model selection, hyperparameter tuning |
| 3 | **Evaluation & Validation** | Cross-validation, MLflow tracking, model registry |
| 4 | Deployment | SageMaker pipelines, automated deployment |
| 5 | CI/CD | Testing, pipeline automation |
| 6 | Monitoring | Drift detection, retraining triggers |

---

# Module 1: Exploratory Data Analysis

---

## Module 1 — Overview

**Goal:** Understand the data before touching a model.

**Stack:**
- Data source: **Amazon Redshift** (UNLOAD to S3 as Parquet)
- Notebook: `eda_example.ipynb`
- Pattern matches the **production pipeline exactly**

```sql
UNLOAD ('SELECT * FROM dth_churn_ml_inference.inference_features')
TO 's3://bucket/path/'
IAM_ROLE '...'
FORMAT AS PARQUET
PARALLEL ON
```

> The same UNLOAD command is used in EDA and in production — no surprises at deploy time.

---

## Module 1 — Key Activities

**1. Data Loading**
- Load from Redshift (production pattern) or local Parquet (fast iteration)
- IAM authentication — no passwords in code

**2. Data Quality Checks**
- Missing values, outliers, column types
- Feature schema validation

**3. Feature Analysis**
- Distributions, correlations, class imbalance
- 155K rows × 58 columns for churn prediction

**4. Train / Val / Test Splits**
- Deterministic, **time-based** partitions — prevents data leakage

---

## Module 1 — Outputs

| Artifact | Purpose |
|----------|---------|
| `feature_schema.json` | Schema definition fed into the ML pipeline |
| `data_metadata.json` | Data provenance for MLOps tracking |
| `train.parquet` / `val.parquet` / `test.parquet` | Versioned, reproducible splits |

---

# Module 2: Model Training

---

## Module 2 — Overview

**Goal:** Develop a reproducible, versioned model through structured experimentation.

**Three-step workflow:**

```
EDA outputs
    ↓
1. Data Preparation  →  cleaned train/val/test
    ↓
2. Model Selection   →  best model + threshold
    ↓
3. Hyperparameter Tuning  →  optimal config + all trials
```

---

## Module 2 — Step 1: Data Preparation

**Apply the same cleaning function to train, val, and test** — never fit on val/test.

| Feature Type | Missing Value Strategy |
|-------------|----------------------|
| Integer | `-1` |
| Numeric | `0.0` |
| Categorical | `'<MISSING>'` |

> Works with any model, gives explicit control, and maintains consistency across environments.

**Output:** `data_preparation_output/` — cleaned Parquet files + `summary.json`

---

## Module 2 — Step 2: Model Selection

**Models compared:** CatBoost vs XGBoost

**Key skill: Threshold Tuning**

- Default threshold (0.5) doesn't give control over recall/precision
- Use `precision_recall_curve()` to find the threshold that hits **80% recall**
- Lower threshold → higher recall, lower precision

**Business context for churn:**
- False negative (missed churner) → lost revenue
- False positive (wrong alert) → wasted retention spend
- Tune to business cost, not just accuracy

---

## Module 2 — Step 3: Hyperparameter Tuning with Optuna

**Optuna TPESampler** — intelligent Bayesian search, not random grid

```python
def objective(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
        "depth":         trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg":   trial.suggest_float("l2", 0.5, 5.0),
    }
    ...
```

**Result for our model:**
`n_estimators=811, lr=0.044, depth=5, l2=1.015` → **ROC-AUC 0.7672**

> More trials = better results, but with diminishing returns. Always optimize on validation set.

---

## Module 2 — Final Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.7672** |
| PR-AUC | 0.6394 |
| Recall (target: 80%) | **80.0%** |
| Precision | 52.7% |
| F1 Score | 0.6353 |
| Threshold | 0.3076 |

**Top features:** `past_churns`, `tipo_produto_actual`, `topup_total_value`

---

# Module 3: Evaluation & Validation

---

## Module 3 — Overview

**Goal:** Validate model robustness and track all experiments with MLflow on SageMaker.

**Key techniques:**
- Cross-validation (temporal splits)
- Feature importance analysis
- Experiment tracking with MLflow

**Infrastructure:** SageMaker MLflow Tracking Server (`dth-churn-ml`, `af-south-1`)

---

## Module 3 — Cross-Validation

**Temporal cross-validation** — respects time ordering of data to prevent leakage.

![bg right:40% 90%](cv_temporal_splits.png)

- Train on past → validate on future
- Multiple folds → robust performance estimate
- Results saved to `cv_results.csv`

> Standard k-fold would leak future data into training — always use time-based splits for time series / sequential data.

---

## Module 3 — MLflow Experiment Tracking

Every training run logs:

| What | How |
|------|-----|
| Hyperparameters | `mlflow.log_param()` |
| Metrics (ROC-AUC, F1…) | `mlflow.log_metric()` |
| Model artifact | `mlflow.log_artifact()` |
| Feature importance | logged as CSV artifact |

```python
mlflow.set_tracking_uri("arn:aws:sagemaker:af-south-1:...")
mlflow.set_experiment("churn-model-v1")
with mlflow.start_run(run_name="catboost_tuned"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({"roc_auc": 0.7672, "recall": 0.80})
```

---

## Module 3 — Quality Gates

Before a model is promoted to the registry it must pass:

| Metric | Minimum |
|--------|---------|
| ROC-AUC | ≥ 0.70 |
| PR-AUC | ≥ 0.50 |
| F1 Score | ≥ 0.50 |
| Recall | ≥ 0.70 |

These same gates are enforced automatically in the **SageMaker Pipeline** (Module 4).

---

# Key MLOps Principles Applied

| Principle | How |
|-----------|-----|
| **Reproducibility** | Fixed seeds, versioned data, documented transformations |
| **Traceability** | All experiments logged in MLflow with metadata |
| **Automation** | Optuna for HPO, SageMaker Pipelines for orchestration |
| **Validation** | Data quality checks before training, quality gates before registration |
| **Production parity** | EDA uses the same UNLOAD pattern as the production pipeline |

---

# What's Next: Module 4 — Deployment

The model is now trained, validated, and registered. Module 4 automates the full lifecycle on AWS:

1. **ExportFromRedshift** — fresh data to S3
2. **TrainChurnModel** — `ml.m5.2xlarge` on SageMaker
3. **EvaluateModel** — compute metrics
4. **CheckModelQuality** — conditional registration if gates pass

```bash
python pipeline.py --create --execute
```

> The pipeline we ran today orchestrates everything learned in Modules 1–3 automatically.
