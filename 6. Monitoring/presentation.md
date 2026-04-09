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
  pre { font-size: 0.8rem; }
  table { font-size: 0.85rem; }
---

# Module 6: Monitoring & Data Drift

### How do you know your model is still working?

---

# The problem: models go stale

You trained a great model. ROC-AUC = 0.77. Recall = 80%. You deployed it.

**Six months later — is it still good?**

The world changes:
- Customers behave differently
- Products change
- Seasonality shifts
- Data pipelines are updated

Your model was trained on **historical data**. If the present looks different from the past, predictions degrade — silently.

> A model can fail without throwing a single error.

---

# A real example

Imagine a churn model trained on data from before a price increase.

After the price increase:
- More customers churn for price-related reasons
- The feature distributions shift
- The model never saw this pattern
- It keeps predicting confidently — but wrongly

Nobody notices until the retention team asks: "why are we losing so many customers we didn't flag?"

---

# Two types of drift

| Type | What changes | Example |
|------|-------------|---------|
| **Data drift** | The input features change | Customers are now older on average |
| **Performance drift** | The model becomes less accurate | Precision drops from 53% to 35% |

They are related but different:
- Data drift can **contribute to** performance drift over time
- You can detect data drift **without labels** (just compare distributions)
- Performance drift requires **ground truth** (you need to know what actually happened)

> Sometimes people call this **model drift**, but **performance drift** is more precise here.

---

# Another useful distinction

For beginners, it also helps to separate **prediction drift** from **performance drift**:

| Concept | What changed? | Need labels? |
|---------|---------------|:---:|
| **Data drift** | Input features changed | No |
| **Prediction drift** | Model scores / output distribution changed | No |
| **Performance drift** | Accuracy, recall, precision, F1, ROC-AUC changed | Yes |

Important idea:
- Prediction drift can be an **early warning**
- But prediction drift alone does **not prove** performance drift

Think of it like a chain:
- Data drift -> may cause -> Prediction drift
- Concept drift -> can cause -> Performance drift
- Prediction drift -> early warning signal
- Performance drift -> where business impact shows up

Here, **concept drift** means the relationship between the inputs and the target has changed.

---

# Data drift — explained simply

**Data drift** = the statistical distribution of your input features has changed compared to when you trained the model.

Think of it like this:

```
Training data (6 months ago):        Live data (today):
  avg age: 32                           avg age: 41
  topup_total_value: ~50                topup_total_value: ~120
  contract_len_days: ~180               contract_len_days: ~90
```

The model was never trained on these patterns. Its predictions are based on a world that no longer exists.

---

# Why data drift is hard to notice

Unlike a bug in your code, drift doesn't cause errors. The model keeps running, returning predictions, looking healthy.

```
SageMaker endpoint: OK
API response time:  120ms
Error rate:         0%

Meanwhile...
  Churn recall:     dropped from 80% to 55%
  False alarm rate: doubled
```

The only way to catch it is to **actively measure it**.

---

# How to detect data drift

Compare the distribution of features **at training time** vs **in production**.

Two common tests:

| Test | What it tells you | Rule of thumb |
|------|-------------------|---------------|
| **PSI** | How much the overall distribution shifted | `< 0.1` stable, `0.1 – 0.25` investigate, `> 0.25` significant |
| **KS test** | Whether the two distributions are significantly different | `p-value < 0.05` suggests drift |

---

# PSI vs KS

- **PSI**: split the feature into bins, compare the % in each bin, then add up the bin-by-bin differences
- **KS**: build the two cumulative distribution curves and take the largest vertical gap between them
- **PSI** = how big is the shift?
- **KS p-value** = is the shift statistically significant?
- Use both together for a better picture

---

# Visualising drift

The simplest way to spot drift is to plot the distribution of a feature at training time vs now.

```
Feature: topup_total_value

Training (blue):  ▓▓▓▓▓▓▓▓▓░░░░░░░░
Production (red): ░░░░░░▓▓▓▓▓▓▓▓▓▓▓

          0    50   100   150   200
```

The whole distribution has shifted right — customers are now topping up more. The model learned patterns from the left side of this chart.

---

# What to monitor

Not every feature needs to be monitored equally. Focus on:

| Priority | What to watch | Why |
|----------|--------------|-----|
| High | Top features by importance | Most impact on predictions |
| High | Prediction score / positive-rate distribution | Is the model output itself shifting? |
| Medium | Missing value rates | Data pipeline issues |
| Medium | Categorical distributions | New categories appearing? |
| Low | Rarely used features | Less impact on model |

For our churn model, start with:
- Numeric drift: `past_churns`, `topup_total_value`, `n_prev_contracts`, `n_dias_subscricao`, `contract_number`
- Categorical drift: `tipo_produto_actual`
- Pipeline checks: `gap_since_prev_expiry`, `topup_avg_value`

---

# Performance drift — how to detect it

Performance drift requires knowing what **actually happened** after prediction.

```
May 1:   model predicts customer X will NOT churn
May 31:  customer X churns   ← ground truth arrives
```

Once you have ground truth, you can compute the same metrics as at training time:
- ROC-AUC, Recall, Precision, F1
- Compare against your training baseline

**The challenge:** ground truth is delayed. For churn, you only know in 30-90 days if the prediction was right.

---

# The monitoring loop

```
                    ┌─────────────────────────────┐
                    │         Production           │
                    │  Model receives live data    │
                    │  Returns predictions         │
                    └──────────┬──────────────────┘
                               │ capture inputs + outputs
                               ↓
                    ┌─────────────────────────────┐
                    │         Monitoring           │
                    │  Compare to baseline         │
                    │  Compute PSI / KS            │
                    │  Track metrics over time     │
                    └──────────┬──────────────────┘
                               │ drift detected
                               ↓
                    ┌─────────────────────────────┐
                    │         Action               │
                    │  Alert the team              │
                    │  Investigate the cause       │
                    │  Retrain if needed           │
                    └─────────────────────────────┘
```

---

# How SageMaker Model Monitor works

SageMaker has a built-in monitoring service called **Model Monitor**.

**What it does:**
1. **Captures** a sample of live requests and responses to S3
2. **Compares** them to a baseline (your training data statistics)
3. **Runs on a schedule** (e.g. daily at 2 AM)
4. **Writes reports** with violations to S3
5. **Publishes metrics** to CloudWatch and can be connected to alarms

You set it up once — it runs automatically from then on.

---

# SageMaker Monitoring Options

| Monitor | What it checks | Needs labels? |
|---------|---------------|:---:|
| **Data Quality** | Feature distributions and schema vs baseline | No |
| **Model Quality** | Accuracy metrics over time | Yes (ground truth) |
| **Bias** | Fairness across selected groups | Usually yes |
| **Explainability** | Feature attribution drift (SHAP) | No |

`Data Quality` and `Model Quality` are part of **Model Monitor**.

`Bias` and `Explainability` are **Clarify-based** monitoring options in SageMaker.

**For beginners: start with Data Quality.** It's the most actionable and doesn't require ground truth.

---

# Setting up Data Quality monitoring

3 things you need:

**1. A baseline** — statistics computed from your training data
```python
# SageMaker generates this automatically from a CSV of your training data
baseline_job = monitor.suggest_baseline(
    baseline_dataset="s3://bucket/baseline.csv",
    dataset_format=DatasetFormat.csv(header=True)
)
```

**2. Data capture** — log a % of live requests to S3
```python
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=20,   # capture 20% of requests
    destination_s3_uri="s3://bucket/capture/"
)
```

**3. A schedule** — run the comparison regularly
```python
monitor.create_monitoring_schedule(
    endpoint_input="my-endpoint",
    schedule_cron_expression="cron(0 2 * * ? *)"
)
```

---

# What a drift report looks like

SageMaker writes a JSON report to S3 after each monitoring run:

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

Each violation tells you which feature drifted and why.

---

# What to do when drift is detected

Not all drift requires immediate action. Use this decision tree:

```
Drift detected
      │
      ├── Is it a data pipeline issue?  ──→  Fix the pipeline
      │   (missing values, wrong types)
      │
      ├── Is model performance still OK? ──→  Monitor more closely
      │   (PSI moderate, metrics stable)
      │
      ├── Has performance degraded?  ──→  Retrain the model
      │   (recall dropped, F1 down)
      │
      └── Is the drift temporary?  ──→  Wait and monitor
          (holiday season, promotion)
```

---

# Retraining — closing the loop

When drift is confirmed, the fix is to retrain on recent data.

With the CI/CD pipeline from Module 5, this is straightforward:

```
Drift detected
      ↓
Update data export query (use more recent dates)
      ↓
git push → GitHub Actions → pipeline.py --create --execute
      ↓
New model trained on recent data
      ↓
Quality gates pass → model registered
      ↓
Deploy new model → update the monitoring baseline if needed
```

This is the **full MLOps loop**: train → deploy → monitor → retrain.

---

# The full MLOps loop

```
  Module 1-2          Module 3            Module 4
  ──────────          ────────            ────────
  EDA + Train    →    Evaluate       →    Deploy to SageMaker
                      (MLflow)            (pipeline.py)
                                               │
  Module 6                               Module 5
  ────────                               ────────
  Drift detected  ←   Monitor       ←   CI/CD automates
  Retrain                                future deploys
```

Each module builds on the previous one. Monitoring is what makes the system **self-sustaining**.

---

# Summary

| Concept | One-line explanation |
|---------|---------------------|
| **Data drift** | Input features look different from training time |
| **Prediction drift** | Model score/output distribution looks different from before |
| **Performance drift** | Accuracy metrics are getting worse once labels arrive |
| **PSI** | How much a distribution shifted (< 0.1 = stable) |
| **KS test** | Compares two distributions; low p-value suggests drift |
| **Baseline** | Statistics from your training data to compare against |
| **Data capture** | Logging a % of live requests to S3 |
| **Model Monitor** | SageMaker service that runs drift checks on a schedule |
| **Retraining trigger** | Drift detected → retrain via CI/CD pipeline |

---

# What's in the notebooks

The notebooks in this folder now cover three different parts of monitoring:

1. `drift_examples.ipynb` - simple feature drift simulations with PSI and KS
2. `prediction_drift_examples.ipynb` - how model scores change after input drift
3. `monitoring_example.ipynb` - SageMaker Model Monitor setup and drift reports

> Start with the drift notebooks, then use `monitoring_example.ipynb` for the SageMaker implementation.

---

# Key takeaways

- Drift is a warning sign, not proof that the model is broken
- Prediction drift is not the same as performance drift
- Predictions can move up or down after drift; both are possible
- Start by monitoring the most important features and basic data quality checks
- Do not retrain automatically; first check whether the change is a data issue, a temporary business shift, or real model degradation
