# Quick Start

## Option 1: Local Training (Start Here!)

The easiest way to get started - train on your machine, data loads from S3.

```bash
# Activate environment
conda activate py312

# Install dependencies
pip install -r requirements.txt

# Train locally
python train.py
```

**What happens:**
- Loads ~110k rows from S3
- Trains CatBoost model
- Saves model to `./models/`

Edit `config.yaml` to change hyperparameters.

---

## Option 2: SageMaker Pipeline

Run the full production pipeline on AWS.

```bash
# Create and execute pipeline
python pipeline.py --create --execute
```

**What happens:**
1. Exports fresh data from Redshift to S3
2. Trains model on ml.m5.2xlarge
3. Evaluates metrics
4. Registers model (if quality gates pass)

Monitor at: https://console.aws.amazon.com/sagemaker/home?region=af-south-1#/pipelines

### Custom Hyperparameters

```bash
python pipeline.py --execute \
  --n-estimators 1000 \
  --learning-rate 0.05
```

---

## Which Option to Choose?

| Option | Best For |
|--------|----------|
| Local Training | Learning, quick experiments, debugging |
| SageMaker Pipeline | Production runs, experiment tracking, team collaboration |

**See README.md for full documentation.**
