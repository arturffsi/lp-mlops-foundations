# Simplified Training Pipeline for SageMaker

This is a beginner-friendly, simplified version of the production training pipeline. It runs on AWS SageMaker but with **much cleaner, easier-to-understand code**.

## What's Different from Production?

### ✅ Simplified (This Version)
- **6 core files** instead of 10+
- **~700 lines** of code instead of 2000+
- Clear, commented functions
- No complex memory management
- No chunking logic
- Straightforward flow
- Production-pattern Redshift export (UNLOAD with IAM)

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
├── data_loader.py       # Simple data loading from S3/Redshift
├── train_utils.py       # Training functions (clean & commented)
├── train.py             # Main training script (~140 lines)
├── evaluate_model.py    # Model evaluation script for pipeline
├── export_redshift.py   # Redshift UNLOAD export (production pattern)
├── pipeline.py          # SageMaker Pipeline orchestration (~350 lines)
├── requirements.txt     # Dependencies
├── .env                 # Redshift password (already configured)
└── README.md           # This file
```

## Quick Start

### Prerequisites

- **AWS credentials** configured (via `aws configure` or environment variables)
- **Python 3.10+** with conda
- **VPN connection** to access Redshift (only needed if reading from Redshift or running pipeline)

**Note:** For local training from S3, you only need AWS credentials - no VPN required!

### 1. Local Testing (Easiest Way to Start!)

```bash
# Activate environment
conda activate py312

# Install dependencies
pip install -r requirements.txt

# Train locally (default: loads data from S3, no VPN needed!)
python train.py

# Or with custom model directory
python train.py --model-dir ./models
```

**What you'll see:**
- Data loading progress (155k rows from S3)
- Training progress with real-time metrics
- Final model performance (ROC-AUC ~0.77)
- Model saved to `./models/` directory

**Takes ~1 minute** to run on a laptop!

### 2. Run as SageMaker Pipeline (Recommended)

**Requirements:**
- VPN connection (for Redshift export step)
- AWS credentials with SageMaker permissions
- Sufficient SageMaker instance quotas (ml.m5.large for processing, ml.m5.2xlarge for training)

**Note:** You may need to update the AWS profile and SageMaker role in `pipeline.py` to match your environment.

```bash
# Create the pipeline in SageMaker
python pipeline.py --create

# Execute the pipeline
python pipeline.py --execute

# Or do both at once
python pipeline.py --create --execute

# With custom hyperparameters
python pipeline.py --execute --n-estimators 1000 --learning-rate 0.05

# With custom instance type
python pipeline.py --execute --instance-type ml.m5.4xlarge
```

### 3. Run as Simple SageMaker Training Job

```python
from sagemaker.pytorch import PyTorch

# Create estimator
estimator = PyTorch(
    entry_point='train.py',
    source_dir='4. Deployment/training_pipeline_simplified',
    role=sagemaker_role,
    instance_type='ml.m5.2xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'n-estimators': 811,
        'learning-rate': 0.044,
        'depth': 5,
        'l2-leaf-reg': 1.015
    }
)

# Train (data loaded from S3 in the script - see config.yaml)
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
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LEARNINGPODS SAGEMAKER PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: ExportFromRedshift                                                  │
│ ═══════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│  ┌──────────────┐                                                          │
│  │   Redshift   │  UNLOAD (IAM Auth)                                       │
│  │   Cluster    │────────────────────┐                                     │
│  │              │                     │                                     │
│  │ prod.dth_    │   • RANDOM() < 0.003 (0.3% sample)                       │
│  │ churn_ml_    │   • svc_sagemaker user                                   │
│  │ training.    │   • RedshiftIAMAuthRole                                  │
│  │ features     │   • FORMAT AS PARQUET                                    │
│  └──────────────┘   • PARALLEL ON                                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────┐                                  │
│  │  S3: learningpods/redshift_export/   │  ~4,800 rows                    │
│  │  *.parquet files                     │  43 features                    │
│  └──────────────────────────────────────┘  target: churn                  │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: TrainChurnModel                                                     │
│ ═══════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│  ┌──────────────────────────────────────┐                                  │
│  │  S3: Parquet files from Step 1      │                                  │
│  └──────────────────────────────────────┘                                  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │  SageMaker Training Job (ml.m5.2xlarge)                 │               │
│  │  ─────────────────────────────────────────────────────  │               │
│  │  • Load data: train.py → data_loader.py                │               │
│  │  • Preprocess: clean types, handle missing values      │               │
│  │  • Split: 80% train / 20% validation                   │               │
│  │  • Train: CatBoost (n_estimators=811, lr=0.044)        │               │
│  │  • Evaluate: ROC-AUC, PR-AUC, F1, Recall, Precision    │               │
│  │  • Log metrics to CloudWatch                           │               │
│  └─────────────────────────────────────────────────────────┘               │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────┐                                  │
│  │  S3: Model artifacts (model.cbm)     │  Training Job Name: XXX          │
│  │      + config.yaml                   │  Metrics logged ✓                │
│  └──────────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: EvaluateModel                                                       │
│ ═══════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│  ┌────────────────────────────────────────┐                                │
│  │  Pipeline Property:                    │                                │
│  │  step_train.properties.TrainingJobName │                                │
│  └────────────────────────────────────────┘                                │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │  SageMaker Processing Job (ml.m5.large)                 │               │
│  │  ─────────────────────────────────────────────────────  │               │
│  │  • evaluate_model.py                                    │               │
│  │  • Call: DescribeTrainingJob API                        │               │
│  │  • Extract: All CloudWatch metrics                      │               │
│  │  • Create: evaluation.json                              │               │
│  │                                                          │               │
│  │  {                                                       │               │
│  │    "metrics": {                                          │               │
│  │      "roc_auc": 0.7672,                                  │               │
│  │      "pr_auc": 0.6394,                                   │               │
│  │      "f1_score": 0.6353,                                 │               │
│  │      "recall": 0.80,                                     │               │
│  │      "precision": 0.527                                  │               │
│  │    }                                                     │               │
│  │  }                                                       │               │
│  └─────────────────────────────────────────────────────────┘               │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────┐                                  │
│  │  S3: evaluation/evaluation.json      │                                  │
│  └──────────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: CheckModelQuality (Conditional)                                     │
│ ═══════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│  ┌──────────────────────────────────────┐                                  │
│  │  Read: evaluation.json               │                                  │
│  └──────────────────────────────────────┘                                  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │  Quality Gates Check                                     │               │
│  │  ─────────────────────────────────────────────────────  │               │
│  │  ✓ ROC-AUC     >= 0.7   ?  (0.7672 ✓)                   │               │
│  │  ✓ PR-AUC      >= 0.5   ?  (0.6394 ✓)                   │               │
│  │  ✓ F1 Score    >= 0.5   ?  (0.6353 ✓)                   │               │
│  │  ✓ Recall      >= 0.7   ?  (0.80   ✓)                   │               │
│  └─────────────────────────────────────────────────────────┘               │
│         │                                                                   │
│         ├─── ALL PASS ──────────┐                                          │
│         │                        │                                          │
│         ▼                        ▼                                          │
│  ┌──────────────┐         ┌────────────────┐                               │
│  │ REGISTER     │         │ SKIP           │                               │
│  │ MODEL TO     │         │ REGISTRATION   │                               │
│  │ MODEL        │         │                │                               │
│  │ REGISTRY     │         │ (Failed gates) │                               │
│  │              │         │                │                               │
│  │ Package:     │         └────────────────┘                               │
│  │ • Model      │                                                           │
│  │ • Metrics    │                                                           │
│  │ • Approval:  │                                                           │
│  │   Pending    │                                                           │
│  └──────────────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────┐                                  │
│  │  Model Package Group:                │                                  │
│  │  learningpods-model-group            │                                  │
│  │                                      │                                  │
│  │  Ready for deployment! 🚀            │                                  │
│  └──────────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY FEATURES                                                                │
│ ─────────────────────────────────────────────────────────────────────────── │
│ • Fully Automated: 4 steps run sequentially                                │
│ • Parameterized: Change hyperparameters without code changes                │
│ • Conditional Registration: Only register if model meets quality standards  │
│ • Production Pattern: IAM auth, UNLOAD, parallel processing                 │
│ • Tracked: All executions visible in SageMaker Console                      │
│ • Reusable: Run multiple times with different parameters                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Data Flow (High Level)

- **Step 1 – ExportFromRedshift**
  Redshift (IAM auth) → UNLOAD command → sampled data (0.3%) → S3 Parquet files.
  Uses `export_redshift.py` with production pattern (no password, IAM role only).

- **Step 2 – TrainChurnModel**
  S3 Parquet → `train.py` → CatBoost training → model artifacts + metrics.
  Metrics include: roc_auc, pr_auc, f1_score, recall, precision, train_samples, valid_samples.

- **Step 3 – EvaluateModel**
  Pipeline passes **exact training job name** → `evaluate_model.py` → DescribeTrainingJob API → `evaluation.json`.
  Extracts all metrics from CloudWatch and prepares for quality gates.

- **Step 4 – CheckModelQuality**
  Reads `evaluation.json` → applies quality gate thresholds → conditional registration.
  Only registers model if ALL quality gates pass (ROC-AUC, PR-AUC, F1, Recall).

**Key Design:** The pipeline passes the exact training job name from Step 2 to Step 3, eliminating any ambiguity about which training job is being evaluated.

### Pipeline Parameters

You can override these when executing:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TrainingInstanceType` | ml.m5.2xlarge | Instance type for training |
| `NEstimators` | 811 | Number of trees |
| `LearningRate` | 0.044 | Learning rate |
| `Depth` | 5 | Tree depth |
| `L2LeafReg` | 1.015 | L2 regularization |
| `RocAucThreshold` | 0.7 | Minimum ROC-AUC for model registration |
| `PrAucThreshold` | 0.5 | Minimum PR-AUC for model registration |
| `F1Threshold` | 0.5 | Minimum F1 Score for model registration |
| `RecallThreshold` | 0.7 | Minimum Recall for model registration |

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

# 6. Evaluate (returns a rich metrics dict)
metrics = evaluate_model(model, X_valid, y_valid, config, train_size=len(y_train))

# 7. Save for SageMaker
save_model(model, config, model_dir)
```

The `metrics` dictionary includes:
- `roc_auc`, `pr_auc`, `f1_score`, `recall`, `precision`, `accuracy`
- `train_samples`: number of rows used for training
- `valid_samples`: number of rows in the validation set
- `feature_count`: number of features used by the model
- Some convenient extra fields (e.g. `actual_churn_rate`, `predicted_churn_rate`)

### That's it! Clean and simple.

## Configuration

Edit `config.yaml` to control training:

```yaml
# Data source (choose one)
data:
  source: "s3"  # Options: "s3", "parquet", or "redshift"
  s3_uri: "s3://your-bucket/data/"  # Used when source = "s3"

# Model hyperparameters
model:
  n_estimators: 811      # Number of trees (from hyperparameter tuning)
  learning_rate: 0.044   # How fast to learn
  depth: 5               # Tree depth
  l2_leaf_reg: 1.015     # Regularization

# Training settings
training:
  test_size: 0.2         # 20% validation split
  target_recall_threshold: 0.8  # Catch 80% of churners
```

## Understanding the Code

### data_loader.py
- `load_from_parquet()`: Loads parquet files from S3 or SageMaker
- `load_from_redshift()`: Connects to Redshift and runs SQL (password-based)
- `load_data()`: Main function that chooses the right loader based on config

### export_redshift.py
- Production-pattern Redshift export using UNLOAD
- IAM authentication (no password needed)
- Uses `redshift_connector` library
- Exports to S3 as Parquet in parallel
- Used in SageMaker Pipeline Step 1

### train_utils.py
- `preprocess_data()`: Cleans data (types, missing values)
- `prepare_features()`: Splits train/test, handles categorical features
- `train_model()`: Trains CatBoost model with early stopping
- `evaluate_model()`: Calculates all metrics (ROC-AUC, PR-AUC, F1, Recall, etc.)
- `save_model()`: Saves for SageMaker deployment
- `write_sagemaker_metrics()`: Writes metrics for CloudWatch

### train.py
- Main training entry point
- Parses command-line arguments
- Orchestrates the training pipeline
- No MLflow (removed for simplicity)

### evaluate_model.py
- Extracts metrics from completed SageMaker training job
- Calls DescribeTrainingJob API
- Writes evaluation.json for quality gates
- Used in SageMaker Pipeline Step 3

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

You can choose where to load data from by editing `config.yaml`:

### Option 1: S3 Parquet Files (Easiest - No VPN!)

**Default option** - pre-exported sample data on S3:

```yaml
data:
  source: "s3"
  s3_uri: "s3://sagemaker-af-south-1-733246370304/learningpods/sample_data/"
```

**Benefits:**
- ✅ No VPN required
- ✅ Fastest option
- ✅ ~155k rows ready to use
- ✅ Works from anywhere

### Option 2: Redshift (Requires VPN)

**Fresh data** directly from Redshift:

```yaml
data:
  source: "redshift"
  redshift_sql: "SELECT * FROM dth_churn_ml_training.training_features WHERE RANDOM() < 0.003;"
  redshift_kwargs:
    host: "redshift-cluster-dsi.cl4o4mmtx9ir.af-south-1.redshift.amazonaws.com"
    port: 5439
    dbname: "prod"
    user: "awsuser"
    password: "${REDSHIFT_PASSWORD}"  # Set in .env file
```

**Requirements:**
- ✅ VPN connection
- ✅ Redshift password in `.env` file: `REDSHIFT_PASSWORD=your_password`
- ✅ Slower (queries database)

**When to use:** When you need the absolute latest data from production.

### Option 3: Export Redshift to S3 (Best of Both Worlds!)

**Pre-export data from Redshift to S3** for fast, VPN-free training:

The easiest way to create your own S3 dataset is to run this UNLOAD command directly in Redshift:

```sql
UNLOAD ('
  SELECT *
  FROM prod.dth_churn_ml_training.training_features
  WHERE RANDOM() < 0.003
')
TO 's3://sagemaker-af-south-1-733246370304/learningpods/sample_data/'
IAM_ROLE 'arn:aws:iam::733246370304:role/RedshiftIAMAuthRole'
FORMAT AS PARQUET
ALLOWOVERWRITE
PARALLEL ON
MANIFEST
REGION 'af-south-1';
```

**Benefits:**
- ✅ One-time export (run on VPN once)
- ✅ Then train anywhere without VPN
- ✅ Fast loading from S3
- ✅ Snapshot of data at a point in time

**When to use:** When you want to share a dataset with others or train repeatedly without VPN.

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
| **Code Lines** | ~700 | ~2000+ |
| **Files** | 6 core files | 10+ files |
| **Complexity** | ⭐⭐ Low-Medium | ⭐⭐⭐⭐⭐ High |
| **SageMaker Pipeline** | ✅ Yes (4 steps) | ✅ Yes (4 steps) |
| **Redshift Export** | ✅ UNLOAD + IAM | ✅ UNLOAD + IAM |
| **Quality Gates** | ✅ Yes | ✅ Yes |
| **Model Registration** | ✅ Conditional | ✅ Conditional |
| **Memory Optimization** | ❌ No | ✅ Yes (chunking) |
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
conda activate py312
pip install -r requirements.txt
```

### "Config file not found"
Make sure `config.yaml` is in the same directory as `train.py`

### "Invalid AWS credentials" or "ClientError: InvalidClientTokenId"
Update the AWS profile in `pipeline.py` (search for `boto3.Session`):
```python
boto_session = boto3.Session(profile_name="YOUR-PROFILE-NAME")
```
Or remove the profile parameter to use default credentials:
```python
boto_session = boto3.Session()  # Uses default credentials
```

### "ResourceLimitExceeded" or "No available instances"
Check your SageMaker instance quotas:
```bash
aws service-quotas list-service-quotas --service-code sagemaker --region af-south-1 | grep -i "ml.m5"
```
Request quota increases if needed, or change instance type in `pipeline.py`.

### "Role does not exist or does not trust sagemaker.amazonaws.com"
Update the role ARN in `pipeline.py` (search for `SageMaker-ExecutionRole`):
```python
role = f"arn:aws:iam::{account_id}:role/service-role/YOUR-SAGEMAKER-ROLE"
```
You can find your SageMaker execution role in the AWS Console:
- Go to IAM → Roles
- Search for "SageMaker"
- Look for a role with "SageMaker" and "ExecutionRole" in the name

### Pipeline execution fails at Redshift export step
- Ensure you're connected to VPN
- Verify Redshift IAM role has S3 write permissions
- Check CloudWatch logs for the processing job

### Model not improving
Try:
- Increase `n_estimators` (more trees)
- Decrease `learning_rate` (slower learning)
- Increase `depth` (deeper trees)
- Check data quality in the Redshift table

## Next Steps

### For Beginners:
1. ✅ **Start with local training** - Run `python train.py` to see the full flow
2. ✅ **Read the training output** - Understand what each metric means
3. ✅ **Experiment with hyperparameters** - Try changing values in `config.yaml`
4. ✅ **Try Redshift data source** - Connect to VPN and switch to `source: "redshift"`

### For Advanced Users:
1. ✅ **Run the SageMaker Pipeline** - Execute `python pipeline.py --create --execute`
2. ✅ **Monitor in AWS Console** - Watch the pipeline steps execute
3. ✅ **Customize quality gates** - Adjust thresholds for model registration
4. ✅ **Graduate to production pipeline** - When you need memory optimization and automation

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
