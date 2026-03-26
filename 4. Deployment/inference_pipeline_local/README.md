# Local Inference Pipeline

A simple, local-only script that uses a trained churn model to predict on new data.

## Prerequisites

- Python 3.10+ with conda
- AWS SSO login (`aws sso login --profile SageMaker-Full-AI-Access-733246370304`)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download the latest model from SageMaker Model Registry
python infer.py --from-registry --input data.parquet --output predictions.csv
```

---

## Where Does the Model Come From?

You have two options:

### Option 1: SageMaker Model Registry (recommended)

Downloads the latest model that passed quality gates in the training pipeline.

```bash
python infer.py --from-registry --input data.parquet
```

This queries the `learningpods-model-group` in SageMaker, downloads `model.tar.gz`, and extracts `model.pkl`, `config.json`, and `metrics.json` to `./models/`.

### Option 2: Local model

Uses the model saved locally by `python train.py` in the training pipeline.

```bash
python infer.py --model-dir ../training_pipeline_simplified/models --input data.parquet
```

---

## Where Does the Data Come From?

### From a file (CSV or Parquet)

```bash
python infer.py --from-registry --input data.parquet
```

### From Redshift (requires VPN)

```bash
python infer.py --from-registry
```

Queries `dth_churn_ml_inference.inference_features` for active subscriptions.

---

## How It Works

```
infer.py (orchestrator)
  │
  ├── Step 1: load_model.py
  │     └── Downloads from SageMaker Registry (--from-registry)
  │     └── Or loads from local models/ folder
  │
  ├── Step 2: load_data.py
  │     └── Loads data from file (CSV/Parquet) or Redshift
  │     └── Preprocesses and prepares features (same as training)
  │
  └── Step 3: predict.py
        └── Runs model.predict_proba() on the data
        └── Applies threshold and saves predictions.csv
```

---

## Output

The output CSV has two columns:

| Column | Description |
|--------|-------------|
| `score` | Probability of churn (0.0 to 1.0) |
| `prediction` | 1 = predicted churner, 0 = not churner |

The threshold comes from `metrics.json` (saved during training). If not found, defaults to 0.5.

---

## File Structure

```
inference_pipeline_local/
├── infer.py          # Entry point — runs all steps
├── load_model.py     # Loads model (from registry or local)
├── load_data.py      # Loads and prepares data for inference
├── predict.py        # Runs predictions and saves output
└── requirements.txt  # Python dependencies
```

---

## Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--from-registry` | Download latest model from SageMaker | Off (uses local model) |
| `--model-dir` | Path to local model directory | `../training_pipeline_simplified/models` |
| `--input` | Input file (.csv or .parquet) | Load from Redshift |
| `--output` | Output CSV path | `predictions.csv` |
