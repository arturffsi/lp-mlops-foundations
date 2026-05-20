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
## Modules 4–6 Summary

**Deployment · CI/CD · Monitoring**

---

# The 6-Week Journey

| Week | Module | Focus |
|------|--------|-------|
| 1 | EDA | Explore data, quality checks, train/val/test splits |
| 2 | Model Training | Data prep, model selection, hyperparameter tuning |
| 3 | Evaluation & Validation | Cross-validation, MLflow tracking, model registry |
| 4 | **Deployment** | SageMaker training pipeline, automated deployment |
| 5 | **CI/CD** | GitHub Actions, automated retraining on push |
| 6 | **Monitoring** | Drift detection, retraining triggers |

---

# Module 4: Deployment

---

## Module 4 — Overview

**Goal:** Move from running code on your laptop to running it automatically on AWS.

**The pipeline has 4 steps:**

```
ExportFromRedshift  →  TrainChurnModel  →  EvaluateModel  →  CheckModelQuality
```

**Key tool:** SageMaker Pipelines — an AWS service that orchestrates these steps as a DAG on managed cloud infrastructure.

---

## Module 4 — Pipeline Steps

| Step | What it does | Instance |
|------|-------------|----------|
| **ExportFromRedshift** | Exports training data to S3 as Parquet | — |
| **TrainChurnModel** | Trains CatBoost on the exported data | `ml.m5.2xlarge` |
| **EvaluateModel** | Computes ROC-AUC, Recall, F1, PR-AUC | `ml.m5.xlarge` |
| **CheckModelQuality** | Registers model only if metrics pass | — |

Steps run in sequence. If any step fails, the pipeline stops.

---

## Module 4 — File Structure

```
training_pipeline_simplified/
├── config.yaml          # Hyperparameters and settings
├── train.py             # Training entry point
├── train_utils.py       # ML logic: preprocessing, training, evaluation
├── data_loader.py       # Loads data from S3 or Redshift
├── pipeline.py          # SageMaker Pipeline definition
├── export_redshift.py   # Step 1: exports data from Redshift
└── evaluate_model.py    # Step 3: computes and writes metrics
```

The same `train.py` runs locally and on SageMaker — no separate version to maintain.

---

## Module 4 — Running the Pipeline

```bash
# Create (upload) the pipeline definition
python pipeline.py --create

# Start a training run
python pipeline.py --execute

# Do both in one command
python pipeline.py --create --execute
```

| Command | What it does |
|---------|-------------|
| `--create` | Packages code and uploads the pipeline DAG to SageMaker |
| `--execute` | Starts a training run using the uploaded definition |
| Need `--create` again? | Yes, every time you change `train.py`, `train_utils.py`, etc. |

---

## Module 4 — Quality Gates

The model is only registered if **all** checks pass:

| Metric | Threshold | Why |
|--------|:---------:|-----|
| ROC-AUC | ≥ 0.70 | Overall model quality |
| PR-AUC | ≥ 0.50 | Performance on imbalanced data |
| F1 Score | ≥ 0.50 | Balance of precision and recall |
| Recall | ≥ 0.70 | % of churners we catch |

If any metric fails → model is **not registered**.
If all pass → model is registered as `PendingManualApproval`.

> Quality gates prevent bad models from reaching production automatically.

---

## Module 4 — Local vs SageMaker

```
LOCAL                            SAGEMAKER
─────                            ─────────
python train.py                  Step 1: export_redshift.py
  ├── config.yaml                  └── Redshift → S3 (Parquet)
  ├── data_loader.py                     ↓
  └── train_utils.py              Step 2: train.py
                                    └── train_utils.py
python pipeline.py --create              ↓
  └── Uploads code to SageMaker   Step 3: evaluate_model.py
                                         ↓
python pipeline.py --execute      Step 4: CheckModelQuality
  └── Starts the pipeline           └── Register if gates pass
```

Start local, scale to SageMaker when ready.

---

# Module 5: CI/CD

---

## Module 5 — Overview

**Goal:** Eliminate the manual step of running `pipeline.py` after every code change.

**Without CI/CD:**
```
You change train.py → you remember to --create → you remember to --execute
→ your teammate doesn't know which version is running in AWS
```

**With CI/CD:**
```
You change train.py → git push → GitHub does --create and --execute automatically
→ everyone can see what ran, when, and whether it succeeded
```

---

## Module 5 — GitHub Actions

GitHub Actions is a built-in GitHub tool that runs commands automatically when you push code.

You configure it with a YAML file in `.github/workflows/`:

```yaml
name: Update & Execute Training Pipeline

on:
  push:
    branches: [main]
    paths:
      - '4. Deployment/training_pipeline_simplified/**'
      - '!4. Deployment/training_pipeline_simplified/*.md'
```

- Triggers on push to `main`
- Only when pipeline files change (not README edits)
- A fresh Ubuntu machine boots, runs the steps, then disappears

---

## Module 5 — What the Workflow Does

```yaml
steps:
  - uses: actions/checkout@v4              # Download code
  - uses: actions/setup-python@v5          # Install Python 3.12
  - run: pip install sagemaker boto3 ...   # Install packages
  - uses: aws-actions/configure-aws-credentials@v4  # Connect to AWS
  - run: python pipeline.py --create       # Upload to SageMaker
  - run: python pipeline.py --execute      # Start training run
```

The runner is a clean machine — it boots, does the job, and is destroyed. Every run is identical and reproducible.

---

## Module 5 — AWS Credentials

The runner needs AWS credentials to talk to SageMaker. **Never put them in the YAML file.**

Store them as **GitHub Secrets** (Settings → Secrets → Actions):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

GitHub encrypts them and injects them at runtime — never visible in logs.

**Who creates these credentials?** An IAM user (`github-actions-sagemaker`) with `AmazonSageMakerFullAccess` + `AmazonS3FullAccess` + `iam:PassRole`.

---

## Module 5 — CI/CD Patterns

| Pattern | When to use | Workflow trigger |
|---------|------------|-----------------|
| **Update only** | Learning / careful teams | `--create` on push |
| **Update + Execute** | Automated retraining | `--create --execute` on push |
| **Scheduled retraining** | Weekly refresh | `cron: '0 8 * * 1'` |
| **PR validation** | Before merging | `--create` on PR open |

Our setup uses **Update + Execute**: every push to `main` that touches pipeline files triggers a full training run.

---

## Module 5 — Visibility

When CI/CD runs, every step is logged in the GitHub Actions tab:

```
✅ Checkout code
✅ Install Python
✅ Install dependencies
✅ Configure AWS credentials
✅ Update pipeline definition
❌ Execute pipeline          ← click to see the full log
```

Much better than "I ran it on my laptop" — everything is visible, logged, and tied to a specific commit.

---

## Module 5 — How CI/CD Connects to Module 4

CI/CD doesn't change anything in SageMaker. It just automates the same two commands you ran manually:

```
git push
    ↓
GitHub Actions
    ├── python pipeline.py --create
    └── python pipeline.py --execute
              ↓
    SageMaker Pipeline
        ├── ExportFromRedshift
        ├── TrainChurnModel
        ├── EvaluateModel
        └── CheckModelQuality
```

The SageMaker pipeline is unchanged — CI/CD is just the trigger.

---

# Module 6: Monitoring

---

## Module 6 — Overview

**Goal:** Detect when the model is no longer performing well in production — before it becomes a business problem.

> A model can fail silently. No errors, no alerts, just wrong predictions.

**The core question:** Is the data the model sees today similar to the data it was trained on?

---

## Module 6 — Why Models Go Stale

A churn model trained 6 months ago learned patterns from that period. If the world has changed:
- A price increase shifts customer behaviour
- New products change usage patterns
- A data pipeline update changes feature values

The model keeps predicting confidently — but on patterns it never saw during training.

---

## Module 6 — Three Types of Drift

| Type | What changes | Example | Need labels? |
|------|-------------|---------|:---:|
| **Data drift** | Input feature distributions change | Customers are now older on average | No |
| **Prediction drift** | The model's output scores shift | Model used to predict 10% churn rate, now predicts 25% | No |
| **Performance drift** | Accuracy metrics degrade | Recall drops from 80% to 55% | Yes |

**Example:** customers start topping up much more than before (data drift)
→ the model now scores 25% of them as likely churners instead of 10% (prediction drift)
→ but are those extra predictions *correct*? You won't know for 30–90 days (performance drift)

- Prediction drift tells you the model's *behaviour* changed — not whether it got *worse*
- Performance drift tells you it actually got worse — but only once ground truth arrives
- You can have prediction drift without performance drift (scores shift, but the model still ranks churners correctly)
- You can have performance drift without obvious prediction drift (scores look stable, but the *meaning* of those scores changed)

---

## Module 6 — Detecting Data Drift

Compare feature distributions at **training time** vs **now**:

| Test | What it measures | Rule of thumb |
|------|-----------------|---------------|
| **PSI** | How much the distribution shifted | `< 0.1` stable · `0.1–0.25` investigate · `> 0.25` significant |
| **KS test** | Whether the two distributions differ significantly | `p-value < 0.05` suggests drift |

Use both together: PSI tells you the magnitude, KS tells you statistical significance.

---

## Module 6 — What to Monitor

| Priority | Features | Why |
|----------|---------|-----|
| High | `past_churns`, `topup_total_value`, `n_prev_contracts` | Top predictors — most impact on model output |
| High | Prediction score distribution | Is the model's output itself shifting? |
| Medium | Missing value rates | Data pipeline issues |
| Medium | `tipo_produto_actual` | Categorical — new values can appear |
| Low | Rarely used features | Less impact |

Monitor the **most important features first**, not everything at once.

---

## Module 6 — SageMaker Model Monitor

SageMaker has a built-in service that automates drift detection:

1. **Data capture** — logs a % of live requests and responses to S3
2. **Baseline** — statistics computed from your training data (`statistics.json`, `constraints.json`)
3. **Monitoring schedule** — runs comparisons on a cron schedule (e.g. every 4h)
4. **Violation reports** — writes JSON to S3 listing which features drifted and why
5. **CloudWatch** — publishes metrics that can trigger alarms

You set it up once — it runs automatically from then on.

---

## Module 6 — Setup: Three Things Needed

**1. Baseline** — statistics from your training data
```python
monitor.suggest_baseline(
    baseline_dataset="s3://bucket/baseline.csv",
    dataset_format=DatasetFormat.csv(header=True)
)
```

**2. Data capture** — log live requests to S3
```python
DataCaptureConfig(enable_capture=True, sampling_percentage=20,
                  destination_s3_uri="s3://bucket/capture/")
```

**3. Monitoring schedule** — run comparisons on a schedule
```python
monitor.create_monitoring_schedule(
    endpoint_input="my-endpoint",
    schedule_cron_expression="cron(0 */4 * * ? *)"  # every 4 hours
)
```

---

## Module 6 — What a Violation Report Looks Like

```json
{
  "violations": [
    {
      "feature_name": "topup_total_value",
      "constraint_check_type": "baseline_drift_check",
      "description": "Feature distribution differs from the baseline"
    },
    {
      "feature_name": "topup_avg_value",
      "constraint_check_type": "completeness_check",
      "description": "Completeness dropped below the allowed threshold"
    }
  ]
}
```

Each entry names the feature, the check that failed, and why.

---

## Module 6 — What to Do When Drift Is Detected

Not all drift requires immediate action:

```
Drift detected
      │
      ├── Data pipeline issue?          →  Fix the pipeline
      │   (missing values, wrong types)
      │
      ├── Performance still OK?         →  Monitor more closely
      │   (PSI moderate, metrics stable)
      │
      ├── Performance degraded?         →  Retrain
      │   (recall dropped, F1 down)
      │
      └── Temporary shift?              →  Wait and monitor
          (holiday season, promotion)
```

> Drift is a warning sign, not proof the model is broken.

---

## Module 6 — Retraining with GitHub Actions

When drift is confirmed, CI/CD from Module 5 handles the retraining:

```
CloudWatch alarm (drift detected)
      ↓
SNS notification
      ↓
Lambda function calls GitHub API
      ↓
GitHub Actions: pipeline.py --create --execute
      ↓
New model trained on recent data
      ↓
Quality gates pass → model registered
      ↓
Update the monitoring baseline
```

This closes the full MLOps loop — **no manual steps required**.

---

# The Full MLOps Loop

```
  Module 1–2           Module 3              Module 4
  ──────────           ────────              ────────
  EDA + Train    →     Evaluate        →     Deploy to SageMaker
                       (MLflow)               (pipeline.py)
                                                    │
  Module 6                                    Module 5
  ────────                                    ────────
  Drift detected  ←    Monitor         ←     CI/CD automates
  → Retrain                                   future deployments
```

Each module builds on the previous one. Monitoring is what makes the system **self-sustaining**.

---

# Key Takeaways — Modules 4–6

| Module | Core idea |
|--------|----------|
| **4 — Deployment** | Same code runs locally and in SageMaker; quality gates prevent bad models reaching production |
| **5 — CI/CD** | `git push` replaces manual `pipeline.py --create --execute`; GitHub Actions is the glue |
| **6 — Monitoring** | Models go stale silently; detect drift early, retrain with the same CI/CD pipeline |

> The last three modules turn a one-off model into a system that trains, deploys, and maintains itself.
