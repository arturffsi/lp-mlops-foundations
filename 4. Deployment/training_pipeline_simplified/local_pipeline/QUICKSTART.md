# Local Pipeline — Quick Start

Runs the same 4-step pipeline as SageMaker, entirely on your machine.

## Run it

```bash
cd 4.\ Deployment/training_pipeline_simplified/local_pipeline

# Default: loads data from S3 (fastest, no VPN needed)
python local_pipeline.py

# Load fresh data from Redshift (requires VPN + .env with REDSHIFT_PASSWORD)
python local_pipeline.py --source redshift

# Override hyperparameters
python local_pipeline.py --n-estimators 1000 --learning-rate 0.05

# Override quality gate thresholds
python local_pipeline.py --roc-auc-threshold 0.65 --recall-threshold 0.7
```

## What happens

| Step | What it does | SageMaker equivalent |
|------|-------------|----------------------|
| 1. Load data | Reads from S3 or Redshift | ExportFromRedshift → TrainingInput |
| 2. Train model | Trains CatBoost locally | TrainChurnModel |
| 3. Evaluate | Computes ROC-AUC, F1, Recall, etc. | EvaluateModel |
| 4. Quality gates | Checks thresholds — saves only if all pass | CheckModelQuality |

## Output

If all quality gates pass, the model is saved to `local_pipeline/models/<timestamp>/` — each run gets its own folder so nothing is overwritten:

```
models/
└── 20260330_081234/
    ├── model.pkl           # Trained CatBoost model
    ├── config.json         # Config used for this run
    ├── hyperparameters.json
    ├── metrics.json        # Full metrics
    └── evaluation.json     # Same format as SageMaker pipeline (for comparison)
```

## Experiment tracking with MLflow

Every run is automatically logged to a local `mlruns/` folder. To view all runs:

```bash
mlflow ui
# then open http://localhost:5000
```

Each run tracks:
- **Params**: hyperparameters, data source, S3 URI / Redshift host, row count
- **Metrics**: ROC-AUC, PR-AUC, F1, Recall, Precision, Accuracy, decision threshold
- **Tags**: `quality_gates_passed`, `data_source`
- **Artifacts**: model files (if gates passed)

This lets you compare runs side by side — useful when experimenting with hyperparameters.

## Quality gate defaults

| Metric | Default threshold |
|--------|:-----------------:|
| ROC-AUC | >= 0.7 |
| PR-AUC | >= 0.5 |
| F1 Score | >= 0.5 |
| Recall | >= 0.7 |
