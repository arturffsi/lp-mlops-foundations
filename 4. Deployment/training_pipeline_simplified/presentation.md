---
marp: true
theme: default
paginate: true
header: "MLOps Foundations — Training Pipeline"
footer: "Learning Pods"
style: |
  section {
    font-size: 28px;
  }
  h1 {
    font-size: 42px;
  }
  h2 {
    font-size: 36px;
  }
  code {
    font-size: 22px;
  }
  table {
    font-size: 22px;
  }
---

<!-- _class: lead -->

# Training Pipeline for SageMaker

### From Local Training to Automated Production Pipelines

---

## What is a Training Pipeline?

A **training pipeline** automates the steps of training a machine learning model:

1. **Get the data** — Export from a database or load from storage
2. **Train the model** — Run the ML algorithm on the data
3. **Evaluate** — Check if the model is good enough
4. **Register** — Save the model for deployment (only if it passes quality checks)

### Why automate it?

- **Reproducibility** — Same steps every time, no human error
- **Scalability** — Train on powerful cloud machines, not your laptop
- **Team collaboration** — Anyone can trigger a training run
- **Auditability** — Every run is tracked with metrics and artifacts

---

## Our Pipeline: 4 Steps

```
┌─────────────────────────────────────────────────────┐
│  STEP 1: ExportFromRedshift                         │
│  Export ~110k rows from Redshift → S3 (Parquet)     │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  STEP 2: TrainChurnModel                            │
│  Train CatBoost model on ml.m5.2xlarge              │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  STEP 3: EvaluateModel                              │
│  Extract metrics (ROC-AUC, Recall, F1, etc.)        │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  STEP 4: CheckModelQuality                          │
│  Register model only if metrics pass quality gates  │
└─────────────────────────────────────────────────────┘
```

---

## File Structure

```
training_pipeline_simplified/
├── config.yaml          # Settings (data source, hyperparameters)
├── train.py             # Main training script (entry point)
├── train_utils.py       # Helper functions for training
├── data_loader.py       # Loads data from S3 or Redshift
├── pipeline.py          # SageMaker Pipeline definition
├── export_redshift.py   # Exports data from Redshift to S3
├── evaluate_model.py    # Extracts metrics for quality gates
└── requirements.txt     # Python dependencies
```

---

## What Each File Does

| File | Purpose | Local? | SageMaker? |
|------|---------|:---:|:---:|
| `train.py` | Entry point — loads config, trains model, saves it | Yes | Yes |
| `train_utils.py` | ML logic: preprocessing, training, evaluation | Yes | Yes |
| `data_loader.py` | Loads data from S3 or Redshift | Yes | Yes |
| `config.yaml` | Hyperparameters and settings for local training | Yes | No |
| `pipeline.py` | Defines the SageMaker Pipeline (4 steps) | Yes (CLI) | No |
| `export_redshift.py` | Exports data from Redshift to S3 (Step 1) | No | Yes |
| `evaluate_model.py` | Extracts metrics for quality gates (Step 3) | No | Yes |

---

## How the Files Connect

```
LOCAL (your laptop)                  SAGEMAKER (AWS cloud)
───────────────────                  ─────────────────────

python train.py                      Step 1: export_redshift.py
  ├── config.yaml                      └── Redshift → S3 (Parquet)
  ├── data_loader.py                         ↓
  └── train_utils.py                 Step 2: train.py
                                       ├── data_loader.py
python pipeline.py --create             └── train_utils.py
  └── Uploads code to SageMaker              ↓
                                     Step 3: evaluate_model.py
python pipeline.py --execute           └── Writes evaluation.json
  └── Starts the pipeline                    ↓
                                     Step 4: CheckModelQuality
                                       └── Register if metrics pass
```

---

## Local Training (Start Here!)

The easiest way to get started — train on your machine:

```bash
# 1. Activate environment
conda activate py312

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train locally (loads data from S3)
python train.py
```

**What happens:**
- Loads ~110k rows from S3
- Trains a CatBoost model
- Saves model to `./models/`
- Prints metrics (ROC-AUC, Recall, F1, etc.)

Edit `config.yaml` to change hyperparameters.

---

## SageMaker Pipeline

Run the full production pipeline on AWS:

```bash
# Create and execute the pipeline
python pipeline.py --create --execute
```

Monitor at: [SageMaker Console](https://console.aws.amazon.com/sagemaker/home?region=af-south-1#/pipelines)

### Custom hyperparameters:

```bash
python pipeline.py --execute \
  --n-estimators 1000 \
  --learning-rate 0.05
```

---

## Understanding `--create` vs `--execute`

### `--create`
Uploads your code and pipeline definition to SageMaker.
**Nothing runs yet** — it just saves the blueprint.

### `--execute`
Starts a training run using the latest uploaded definition.

### When do you need `--create`?

| Change | Need `--create`? |
|--------|:---:|
| Changed `train.py`, `train_utils.py`, etc. | Yes |
| Changed hyperparameter defaults | Yes |
| Just want to re-run with same code | No, just `--execute` |
| Override hyperparameters at runtime | No, pass as `--execute` params |

---

## Quality Gates

The pipeline only registers the model if it passes **all** quality checks:

| Metric | Threshold | What it Means |
|--------|:---------:|---------------|
| ROC-AUC | >= 0.7 | Overall model quality |
| PR-AUC | >= 0.5 | Performance on imbalanced data |
| F1 Score | >= 0.5 | Balance of precision and recall |
| Recall | >= 0.7 | % of churners we catch |

If **any** metric fails → model is **not registered**.
If **all** pass → model is registered as `PendingManualApproval`.

This prevents bad models from reaching production.

---

## Understanding the Metrics

```
ROC-AUC:    0.77    # Model quality (0.5 = random, 1.0 = perfect)
Recall:     80%     # % of churners we catch
Precision:  53%     # % of predictions that are correct
F1 Score:   0.64    # Balance of recall and precision
```

### What's a good score?

- **ROC-AUC:** 0.7+ is good, 0.8+ is very good
- **Recall:** Higher = catch more churners (but more false alarms)
- **Precision:** Higher = fewer false alarms (but miss more churners)
- **F1:** Harmonic mean — balances the trade-off

**For churn prediction, we prioritize Recall** — better to flag a false alarm than miss a real churner.

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| **Pipeline** | 4 steps: Export → Train → Evaluate → Register |
| **Local training** | `python train.py` — start here for learning |
| **SageMaker** | `python pipeline.py --create --execute` |
| **`--create`** | Uploads code to SageMaker (re-run after code changes) |
| **`--execute`** | Starts a training run |
| **Quality gates** | Model only registered if metrics pass thresholds |

---

<!-- _class: lead -->

## Next Steps

1. **Run local training** — `python train.py`
2. **Experiment with hyperparameters** — edit `config.yaml`
3. **Run the SageMaker Pipeline** — `python pipeline.py --create --execute`
4. **Monitor in AWS Console** — watch the pipeline steps execute
