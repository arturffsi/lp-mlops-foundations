# Quick Start (5 Minutes)

## Recommended: Run on SageMaker Pipeline

This is the easiest way - SageMaker handles all AWS authentication automatically!

```bash
# Activate environment
conda activate mlcourse

# Install dependencies
pip install -r requirements.txt

# Create and execute pipeline
python pipeline.py --create --execute
```

**That's it!** The pipeline will:
1. Export data from Redshift to S3
2. Train the model on SageMaker
3. Evaluate metrics
4. Register model (if quality gates pass)

Monitor at: https://console.aws.amazon.com/sagemaker/home?region=af-south-1#/pipelines

---

## Customization

```bash
# Custom hyperparameters
python pipeline.py --execute \
  --n-estimators 1000 \
  --learning-rate 0.05

# Stricter quality gates
python pipeline.py --execute \
  --roc-auc-threshold 0.75 \
  --recall-threshold 0.8
```

---

## What's Configured

- **Data**: Redshift → Exports to S3 automatically in the pipeline
- **Model**: CatBoost with optimized hyperparameters (n_estimators=811, lr=0.044)
- **Quality Gates**: ROC-AUC≥0.7, PR-AUC≥0.5, F1≥0.5, Recall≥0.7
- **Sample Size**: 0.3% of data (fast training for learning)

**See README.md for full documentation.**

---

## Why SageMaker?

Running on SageMaker instead of locally:
- ✅ **Automatic AWS authentication** - no credential headaches
- ✅ **Scalable compute** - use powerful instances (ml.m5.xlarge)
- ✅ **Experiment tracking** - all runs logged automatically
- ✅ **Production ready** - same infrastructure as production ML
- ✅ **Pipeline orchestration** - reproducible workflows

**Note**: Local execution is possible but requires complex AWS profile setup. For beginners, always use SageMaker!
