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

### `--create` vs `--execute`

| Flag | What it does | When to use |
|------|-------------|-------------|
| `--create` | Uploads (or overwrites) the pipeline definition (DAG) to SageMaker | First time, or after changing `pipeline.py` |
| `--execute` | Starts a new pipeline run using the uploaded definition | Every time you want to train |

- **First time:** use both together — `--create --execute`
- **Subsequent runs:** `--execute` alone is enough (the definition is already uploaded)
- **After editing `pipeline.py`:** use `--create --execute` again to push the updated definition
- **Running `--create` on an existing pipeline is safe** — it does an upsert (overwrites the definition without affecting past runs or registered models)

```bash
# First time (or after changing pipeline.py)
python pipeline.py --create --execute

# Subsequent runs
python pipeline.py --execute
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
