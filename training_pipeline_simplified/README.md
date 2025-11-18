# Simplified Training Pipeline for SageMaker

This is a beginner-friendly, simplified version of the production training pipeline. It runs on AWS SageMaker but with **much cleaner, easier-to-understand code**.

## What's Different from Production?

### ✅ Simplified (This Version)
- **3 files** instead of 10+
- **~500 lines** of code instead of 2000+
- Clear, commented functions
- No complex memory management
- No chunking logic
- Straightforward flow

### ⚡ Production Version
- Complex memory optimization
- Chunked data processing
- Multiple pipeline orchestration files
- Advanced error handling
- More configuration options

## File Structure

```
training_pipeline_simplified/
├── config.yaml          # All settings in one place
├── data_loader.py       # Simple data loading (Redshift)
├── train_utils.py       # Training functions (clean & commented)
├── train.py             # Main training script (~150 lines)
├── pipeline.py          # SageMaker Pipeline orchestration (~200 lines)
├── requirements.txt     # Dependencies
├── .env                 # Redshift password (already configured)
└── README.md           # This file
```

## Quick Start

### Prerequisites

- AWS credentials configured (via `aws configure` or environment variables)
- VPN connection to access Redshift (only if using `source: "redshift"` in config)
- Python 3.8+ with conda

### 1. Local Testing

```bash
# Activate environment
conda activate mlcourse

# Install dependencies
pip install -r requirements.txt

# Train locally (default: loads data from S3, no VPN needed!)
python train.py --mlflow-mode disabled
```

### 2. Run as SageMaker Pipeline (Recommended)

```bash
# Create the pipeline in SageMaker
python pipeline.py --create

# Execute the pipeline
python pipeline.py --execute

# Or do both at once
python pipeline.py --create --execute

# With custom hyperparameters
python pipeline.py --execute --n-estimators 1000 --learning-rate 0.05
```

### 3. Run as Simple SageMaker Training Job

```python
from sagemaker.pytorch import PyTorch

# Create estimator
estimator = PyTorch(
    entry_point='train.py',
    source_dir='training_pipeline_simplified',
    role=sagemaker_role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'n-estimators': 811,
        'learning-rate': 0.044,
        'depth': 5,
        'mlflow-mode': 'disabled'
    }
)

# Train (data loaded from Redshift in the script)
estimator.fit()
```

## SageMaker Pipeline

The `pipeline.py` creates a **SageMaker Pipeline** named `learningpods-pipeline` that orchestrates the training workflow.

### What is a SageMaker Pipeline?

A SageMaker Pipeline is like a recipe that defines the steps needed to train and deploy a model. Benefits:
- **Repeatable**: Run the same workflow multiple times with different parameters
- **Trackable**: See all executions and their results in AWS Console
- **Parameterized**: Change hyperparameters without modifying code
- **Scalable**: Runs on AWS infrastructure, not your laptop

### Pipeline Structure

```
learningpods-pipeline
│
├── 1. ExportFromRedshift
│   ├── Connects to Redshift
│   ├── Exports 0.3% sample to S3
│   └── Saves as Parquet
│
├── 2. TrainChurnModel
│   ├── Loads data from S3
│   ├── Trains CatBoost model
│   ├── Evaluates on validation set
│   └── Saves model artifacts
│
├── 3. EvaluateModel
│   ├── Extracts metrics from training job
│   └── Prepares evaluation.json
│
└── 4. CheckModelQuality (Conditional)
    ├── Checks if metrics pass quality gates
    ├── If PASS: Registers model to Model Registry
    └── If FAIL: Skips registration
```

### Pipeline Parameters

You can override these when executing:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TrainingInstanceType` | ml.m5.xlarge | Instance type for training |
| `NEstimators` | 811 | Number of trees |
| `LearningRate` | 0.044 | Learning rate |
| `Depth` | 5 | Tree depth |
| `L2LeafReg` | 1.015 | L2 regularization |

### Example Commands

```bash
# Create pipeline with default settings
python pipeline.py --create

# Execute with custom hyperparameters
python pipeline.py --execute \
  --n-estimators 1000 \
  --learning-rate 0.05 \
  --depth 6

# Use a larger instance
python pipeline.py --execute --instance-type ml.m5.2xlarge
```

### Monitoring Pipeline Execution

After starting a pipeline execution, you can monitor it:

1. **AWS Console**: https://console.aws.amazon.com/sagemaker/home?region=af-south-1#/pipelines
2. **Via Code**:
   ```python
   # Get execution status
   execution.describe()

   # List all steps
   execution.list_steps()

   # Wait for completion
   execution.wait()
   ```

## How It Works

### Step-by-Step Flow

```python
# 1. Load configuration
config = load_config('config.yaml')

# 2. Load data (from S3 or Redshift)
df = load_data(config)

# 3. Preprocess (clean data types, handle missing values)
df = preprocess_data(df, config)

# 4. Prepare features (split train/test, identify categorical features)
X_train, X_valid, y_train, y_valid, cat_indices = prepare_features(df, config)

# 5. Train model
model = train_model(X_train, X_valid, y_train, y_valid, cat_indices, config)

# 6. Evaluate
metrics = evaluate_model(model, X_valid, y_valid, config)

# 7. Save for SageMaker
save_model(model, config, model_dir)
```

### That's it! Clean and simple.

## Configuration

Edit `config.yaml` to control training:

```yaml
# Data source
data:
  source: "parquet"  # or "redshift"
  parquet_uri: "s3://your-bucket/data/"

# Model hyperparameters
model:
  n_estimators: 500      # Number of trees
  learning_rate: 0.1     # How fast to learn
  depth: 5               # Tree depth
  target_recall_threshold: 0.8  # Catch 80% of churners
```

## Understanding the Code

### data_loader.py
- `load_from_parquet()`: Loads parquet files from S3 or SageMaker
- `load_from_redshift()`: Connects to Redshift and runs SQL
- `load_data()`: Main function that chooses the right loader

### train_utils.py
- `preprocess_data()`: Cleans data (types, missing values)
- `prepare_features()`: Splits train/test, handles categorical features
- `train_model()`: Trains CatBoost model
- `evaluate_model()`: Calculates and prints all metrics
- `save_model()`: Saves for SageMaker deployment
- `write_sagemaker_metrics()`: Writes metrics for hyperparameter tuning

### train.py
- Main entry point
- Parses arguments
- Calls functions in order
- Handles MLflow logging (optional)

## Hyperparameter Tuning

You can override any hyperparameter from the command line:

```bash
python train.py \
  --n-estimators 1000 \
  --learning-rate 0.05 \
  --depth 6 \
  --l2-leaf-reg 5.0
```

Or in SageMaker:

```python
hyperparameters = {
    'n-estimators': 1000,
    'learning-rate': 0.05,
    'depth': 6,
    'l2-leaf-reg': 5.0
}

estimator.set_hyperparameters(**hyperparameters)
```

## Data Sources

### Option 1: Parquet Files (Easiest)

```yaml
data:
  source: "parquet"
  parquet_uri: "s3://your-bucket/data/"
```

### Option 2: Redshift

```yaml
data:
  source: "redshift"
  redshift_sql: "SELECT * FROM your_table"
  redshift_kwargs:
    host: "your-cluster.redshift.amazonaws.com"
    port: 5439
    dbname: "your_db"
    user: "your_user"
    password: "${REDSHIFT_PASSWORD}"  # From environment
```

## Metrics Explained

After training, you'll see:

```
📈 VALIDATION RESULTS
===============================
ROC-AUC:              0.8542    # Overall model quality (0.5-1.0)
PR-AUC:               0.6234    # Precision-Recall AUC

Target Recall:        80.0%     # We wanted to catch 80% of churners
Actual Recall:        82.5%     # We actually caught 82.5%!
Precision:            45.3%     # Of predicted churners, 45.3% really churn
F1 Score:             0.5841    # Balance of precision & recall
Accuracy:             88.2%     # Overall correct predictions
Threshold:            0.3214    # Probability cutoff we use
```

### What's a Good Score?

- **ROC-AUC**: 0.7-0.8 = Good, 0.8-0.9 = Very Good, 0.9+ = Excellent
- **Recall**: Higher is better (catching more churners)
- **Precision**: Higher is better (fewer false alarms)
- **F1**: Balance between recall and precision

## Common Tweaks

### Want Better Performance?
```yaml
model:
  n_estimators: 1000     # More trees
  learning_rate: 0.05    # Slower learning
  depth: 6               # Deeper trees
```

### Want Faster Training?
```yaml
model:
  n_estimators: 200      # Fewer trees
  learning_rate: 0.15    # Faster learning
```

### Want to Catch More Churners?
```yaml
training:
  target_recall_threshold: 0.9  # Catch 90% instead of 80%
```

## Comparison to Production

| Feature | Simplified | Production |
|---------|-----------|-----------|
| **Code Lines** | ~500 | ~2000+ |
| **Files** | 3 | 10+ |
| **Complexity** | ⭐ Low | ⭐⭐⭐⭐⭐ High |
| **SageMaker** | ✅ Yes | ✅ Yes |
| **MLflow** | ✅ Basic | ✅ Advanced |
| **Memory Optimization** | ❌ No | ✅ Yes (chunking, etc) |
| **Pipeline Orchestration** | ❌ No | ✅ Yes |
| **Hyperparameter Tuning** | Manual | ✅ Automated |
| **Best For** | Learning | Production |

## When to Use Each

### Use Simplified Pipeline When:
- ✅ Learning MLOps
- ✅ Quick experiments
- ✅ Dataset < 10M rows
- ✅ Prototyping
- ✅ Teaching others

### Use Production Pipeline When:
- ✅ Deploying to production
- ✅ Dataset > 10M rows
- ✅ Need memory optimization
- ✅ Need automated pipelines
- ✅ Team collaboration

## Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Config file not found"
Make sure `config.yaml` is in the same directory as `train.py`

### "No data found in /opt/ml/input/data/training"
Check that you passed data to SageMaker:
```python
estimator.fit({'training': 's3://your-bucket/data/'})
```

### Model not improving
Try:
- Increase `n_estimators`
- Decrease `learning_rate`
- Check for data quality issues

## Next Steps

1. **Run it locally** first to understand the flow
2. **Experiment** with different hyperparameters
3. **Deploy to SageMaker** when ready
4. **Graduate** to the production pipeline when you need more features

## Key Learning Points

By working with this simplified pipeline, you'll learn:

✅ How to structure ML code cleanly
✅ How SageMaker training works
✅ How to load data from different sources
✅ How to prepare features for training
✅ How to train and evaluate models
✅ How to tune hyperparameters
✅ How to save models for deployment

Happy learning! 🚀

---

**Questions?** Compare with `../training_pipeline/` to see the production version.
