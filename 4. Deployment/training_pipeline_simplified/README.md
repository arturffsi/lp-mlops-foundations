# Simplified Training Pipeline for SageMaker

A beginner-friendly ML training pipeline that runs on AWS SageMaker.

## Quick Start

### Prerequisites

- Python 3.10+ with conda
- AWS credentials configured (`aws configure sso`)

### 1. Local Training (Start Here!)

```bash
# Activate environment
conda activate py312

# Install dependencies
pip install -r requirements.txt

# Train locally (loads data from S3)
python train.py
```

**What you'll see:**
- Data loading (~110k rows from S3)
- Training progress with metrics
- Final model saved to `./models/`

### 2. Run as SageMaker Pipeline

```bash
# Create and execute the pipeline
python pipeline.py --create --execute
```

Monitor at: https://console.aws.amazon.com/sagemaker/home?region=af-south-1#/pipelines

---

## Pipeline Overview

The pipeline has **4 steps**:

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: ExportFromRedshift                                 │
│  Export ~110k rows from Redshift → S3 (Parquet)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: TrainChurnModel                                    │
│  Train CatBoost model on ml.m5.2xlarge                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: EvaluateModel                                      │
│  Extract metrics (ROC-AUC, Recall, F1, etc.)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: CheckModelQuality                                  │
│  Register model only if metrics pass quality gates          │
│  (ROC-AUC >= 0.7, Recall >= 0.7, F1 >= 0.5)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

Edit `config.yaml` to change settings for **local training** (`python train.py`):

> **Note:** The SageMaker Pipeline ignores `config.yaml` - it has its own settings hardcoded in `pipeline.py`.

```yaml
# Data source
data:
  source: "s3"  # Options: "s3" or "redshift"
  s3_uri: "s3://sagemaker-af-south-1-733246370304/learningpods/sample_data/"

# Model hyperparameters
model:
  n_estimators: 811    # Number of trees
  learning_rate: 0.044 # How fast to learn
  depth: 5             # Tree depth

# Training settings
training:
  test_size: 0.2                   # 20% for validation
  target_recall_threshold: 0.8     # Catch 80% of churners
```

### Data Sources

| Source | When to Use | VPN Required? |
|--------|-------------|---------------|
| `s3` | Default - fastest, pre-exported data | No |
| `redshift` | Need fresh data from database | Yes (local only) |

**Note:** The SageMaker Pipeline always exports fresh data from Redshift (no VPN needed - it runs inside AWS).

---

## Understanding the Metrics

After training, you'll see:

```
ROC-AUC:    0.77    # Model quality (0.5-1.0, higher is better)
Recall:     80%     # % of churners we catch
Precision:  53%     # % of predictions that are correct
F1 Score:   0.64    # Balance of recall and precision
```

**What's a good score?**
- ROC-AUC: 0.7+ is good, 0.8+ is very good
- Recall: Higher = catch more churners (but more false alarms)
- Precision: Higher = fewer false alarms (but miss more churners)

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
├── requirements.txt     # Python dependencies
└── QUICKSTART.md        # Quick start guide
```

### What Each File Does

| File | Purpose | Runs locally? | Runs on SageMaker? |
|------|---------|:---:|:---:|
| **`train.py`** | Entry point for training. Loads config, calls data loader and training utils, saves the model. This is the script SageMaker executes inside the training container. | Yes | Yes |
| **`train_utils.py`** | All the ML logic: preprocessing, feature preparation, CatBoost training, evaluation, and model saving. Separated from `train.py` to keep things organized. | Yes | Yes |
| **`data_loader.py`** | Handles loading data from different sources (S3 parquet files or Redshift). Automatically detects if it's running locally or inside SageMaker. | Yes | Yes |
| **`config.yaml`** | Configuration for local training: data source, hyperparameters, feature lists, and training settings. **Not used by the SageMaker Pipeline** (pipeline has its own config in `pipeline.py`). | Yes | No |
| **`pipeline.py`** | Defines the SageMaker Pipeline DAG (the 4 steps). Use `--create` to upload the definition and `--execute` to start a run. Only runs on your laptop — it talks to the SageMaker API. | Yes (CLI) | No |
| **`export_redshift.py`** | Connects to Redshift using IAM auth and runs an UNLOAD command to export data as Parquet to S3. Runs inside a SageMaker processing container (Step 1). | No | Yes |
| **`evaluate_model.py`** | Extracts metrics from the completed training job and writes them to `evaluation.json`. SageMaker uses this file to decide if the model should be registered (Step 3). | No | Yes |
| **`requirements.txt`** | Python packages needed for local training (`catboost`, `pyarrow`, etc.). SageMaker containers have their own dependencies. | Yes | No |

### How the Files Connect

```
LOCAL (your laptop)                    SAGEMAKER (AWS cloud)
───────────────────                    ─────────────────────

python train.py                        Step 1: export_redshift.py
  ├── config.yaml                        └── Redshift → S3 (Parquet)
  ├── data_loader.py                           ↓
  └── train_utils.py                   Step 2: train.py
                                         ├── data_loader.py
python pipeline.py --create               └── train_utils.py
  └── Uploads code + DAG to SageMaker          ↓
                                       Step 3: evaluate_model.py
python pipeline.py --execute             └── Writes evaluation.json
  └── Starts the pipeline                      ↓
                                       Step 4: CheckModelQuality
                                         └── Register if metrics pass
```

---

## Customizing Hyperparameters

### Via Command Line (Local)

```bash
python train.py --n-estimators 1000 --learning-rate 0.05 --depth 6
```

### Via Pipeline Parameters

```bash
python pipeline.py --execute --n-estimators 1000 --learning-rate 0.05
```

---

## Troubleshooting

### "Module not found"
```bash
conda activate py312
pip install -r requirements.txt
```

### "Invalid AWS credentials"
```bash
aws sso login --profile SageMaker-Full-AI-Access-733246370304
```

### "Config file not found"
Make sure you're running from the `training_pipeline_simplified/` directory.

### Pipeline fails at Redshift export
- Check that `SageMaker-ExecutionRole` has `AmazonEC2ContainerRegistryReadOnly` permission
- Check CloudWatch logs for the processing job

---

## Next Steps

1. **Run local training** - `python train.py`
2. **Experiment with hyperparameters** - Edit `config.yaml`
3. **Run the SageMaker Pipeline** - `python pipeline.py --create --execute`
4. **Monitor in AWS Console** - Watch the pipeline steps execute
