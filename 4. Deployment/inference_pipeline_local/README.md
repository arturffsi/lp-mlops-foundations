# Simple Local Inference (Beginner)

This is a minimal, local-only inference script for the churn model trained in
`training_pipeline_simplified`.

## Run (local file)

```
python infer.py \
  --input /path/to/input.parquet \
  --model-dir ../training_pipeline_simplified/models \
  --output predictions.csv
```

## Run (load from S3/Redshift using config)

This uses the same data loading as the training pipeline (`config.yaml`), but
defaults to Redshift and runs **today-only** data from:
`dth_churn_ml_inference.inference_features`.

It auto-picks a date/timestamp column (preferring names like `snapshot_date`,
`dt_ref`, etc.) and filters to `current_date`.

```
python infer.py \
  --model-dir ../training_pipeline_simplified/models \
  --output predictions.csv
```

## What it does

1. Loads a saved model (`model.pkl` or `model.cbm`)
2. Loads input data from CSV/Parquet *or* from S3/Redshift via config
3. Applies the same basic preprocessing as training
4. Predicts and saves `score` + `prediction`

## Output

- `score` = probability of churn (if model supports `predict_proba`)
- `prediction` = 1 if score >= threshold (from `metrics.json` or default 0.5)
