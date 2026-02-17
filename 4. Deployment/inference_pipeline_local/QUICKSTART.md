# Quick Start

```
pip install -r requirements.txt
```

```
python infer.py \
  --input /path/to/input.parquet \
  --model-dir ../training_pipeline_simplified/models \
  --output predictions.csv
```

## Or load from S3/Redshift (uses config.yaml)

```
python infer.py \
  --model-dir ../training_pipeline_simplified/models \
  --output predictions.csv
```

Default query (today only):
`SELECT * FROM dth_churn_ml_inference.inference_features WHERE <date_col>::date = current_date;`
