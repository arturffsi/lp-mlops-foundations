"""
Local Training Pipeline for Churn Prediction

A local equivalent of the SageMaker pipeline — same 4 steps, runs entirely on your machine.
No AWS compute needed. Models are saved locally.

Steps:
  1. Load data      — from S3 or Redshift (your choice)
  2. Train model    — CatBoost on your machine
  3. Evaluate model — compute metrics on validation set
  4. Quality gates  — only save model if metrics pass thresholds

Usage:
  python local_pipeline.py                        # Run with default config (S3)
  python local_pipeline.py --source redshift      # Load fresh data from Redshift
  python local_pipeline.py --source s3            # Load from S3 (default, fastest)

  # Override hyperparameters
  python local_pipeline.py --n-estimators 1000 --learning-rate 0.05

  # Override quality gate thresholds
  python local_pipeline.py --roc-auc-threshold 0.75 --recall-threshold 0.8
"""

import argparse
import json
import os
import sys
import yaml
from datetime import datetime
import mlflow

# Add the parent directory to the path so we can import the shared scripts
# (data_loader.py, train_utils.py all live one level up)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import load_data, load_from_redshift, load_from_parquet
from train_utils import preprocess_data, prepare_features, train_model, evaluate_model, save_model


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Step 1: Load Data
# =============================================================================

def step_load_data(config, source_override=None):
    """
    Step 1: Load data from S3 or Redshift.

    This mirrors the ExportFromRedshift + TrainingInput steps in the SageMaker pipeline.
    Here we just load the data directly — no S3 export needed.

    Args:
        config: Configuration dictionary
        source_override: 's3' or 'redshift' — overrides config['data']['source']

    Returns:
        DataFrame with loaded data
    """
    print("\n" + "="*60)
    print("STEP 1: Load Data")
    print("="*60)

    # Override source if provided via CLI
    if source_override:
        config['data']['source'] = source_override
        print(f"   Source overridden to: {source_override}")

    source = config['data']['source']

    if source == 'redshift':
        # Load fresh data directly from Redshift.
        # Requires VPN + REDSHIFT_PASSWORD in your .env file.
        print("   Source: Redshift (fresh data, requires VPN)")
        df = load_from_redshift(
            config['data']['redshift_sql'],
            config['data']['redshift_kwargs']
        )
    elif source == 's3':
        # Load from the pre-exported parquet files in S3.
        # Fastest option — no VPN needed, just AWS CLI authenticated.
        print("   Source: S3 (pre-exported parquet files)")
        df = load_from_parquet(config['data']['s3_uri'])
    elif source == 'parquet':
        # Load from a local or S3 parquet file.
        print("   Source: Parquet")
        df = load_from_parquet(config['data']['parquet_uri'])
    else:
        raise ValueError(f"Unknown source: {source}. Use 's3', 'redshift', or 'parquet'")

    print(f"\nStep 1 complete — loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


# =============================================================================
# Step 2: Train Model
# =============================================================================

def step_train(df, config):
    """
    Step 2: Preprocess data and train the CatBoost model.

    This mirrors the TrainChurnModel step in the SageMaker pipeline.
    The same train.py / train_utils.py logic runs here, just locally.

    Args:
        df: Raw DataFrame from Step 1
        config: Configuration dictionary

    Returns:
        model, X_valid, y_valid — the trained model and validation set for evaluation
    """
    print("\n" + "="*60)
    print("STEP 2: Train Model")
    print("="*60)

    # Preprocess: type conversions, fill nulls
    df = preprocess_data(df, config)

    # Prepare features: drop IDs, split train/val, identify categoricals
    X_train, X_valid, y_train, y_valid, cat_indices = prepare_features(df, config)

    # Train CatBoost
    model = train_model(X_train, X_valid, y_train, y_valid, cat_indices, config)

    print(f"\nStep 2 complete — model trained ({model.best_iteration_} best iteration)")
    return model, X_train, X_valid, y_train, y_valid


# =============================================================================
# Step 3: Evaluate Model
# =============================================================================

def step_evaluate(model, X_train, X_valid, y_train, y_valid, config):
    """
    Step 3: Compute evaluation metrics on the validation set.

    This mirrors the EvaluateModel step in the SageMaker pipeline.
    Instead of writing evaluation.json to /opt/ml/processing/evaluation,
    we return the metrics dict and save it locally.

    Args:
        model: Trained CatBoost model
        X_valid, y_valid: Validation data
        config: Configuration dictionary

    Returns:
        metrics dict (roc_auc, pr_auc, f1_score, recall, etc.)
    """
    print("\n" + "="*60)
    print("STEP 3: Evaluate Model")
    print("="*60)

    metrics = evaluate_model(model, X_valid, y_valid, config, train_size=len(y_train))

    print(f"\nStep 3 complete — metrics computed")
    return metrics


# =============================================================================
# Step 4: Quality Gates
# =============================================================================

def step_quality_gates(metrics, thresholds):
    """
    Step 4: Check if the model passes all quality thresholds.

    This mirrors the CheckModelQuality ConditionStep in the SageMaker pipeline.
    If ALL conditions pass → save the model.
    If ANY condition fails → skip saving.

    Args:
        metrics: dict from step_evaluate
        thresholds: dict with keys roc_auc, pr_auc, f1_score, recall

    Returns:
        passed (bool): True if all gates passed
    """
    print("\n" + "="*60)
    print("STEP 4: Quality Gates")
    print("="*60)

    # Define which metrics to check and their thresholds
    gates = [
        ("roc_auc",   thresholds['roc_auc'],   "ROC-AUC"),
        ("pr_auc",    thresholds['pr_auc'],     "PR-AUC"),
        ("f1_score",  thresholds['f1_score'],   "F1 Score"),
        ("recall",    thresholds['recall'],     "Recall"),
    ]

    all_passed = True
    for metric_key, threshold, label in gates:
        value = metrics[metric_key]
        passed = value >= threshold
        status = "PASS" if passed else "FAIL"
        print(f"   {status}  {label}: {value:.4f} (threshold: {threshold})")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All quality gates passed — model will be saved")
    else:
        print("One or more quality gates failed — model will NOT be saved")

    print(f"\nStep 4 complete")
    return all_passed


# =============================================================================
# Save model and evaluation locally
# =============================================================================

def save_outputs(model, config, metrics, model_dir):
    """
    Save model artifacts and evaluation results locally.

    In the SageMaker pipeline this is done by RegisterModel.
    Here we just write everything to a local directory.

    Saves:
      - model.pkl
      - config.json
      - hyperparameters.json
      - metrics.json
      - evaluation.json  (same format as SageMaker pipeline uses for quality gates)
    """
    print(f"\nSaving model artifacts to: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model + config + hyperparameters + metrics (reuses existing save_model function)
    save_model(model, config, model_dir, metrics=metrics)

    # Also save evaluation.json in the same format the SageMaker pipeline uses.
    # Useful if you want to compare local vs cloud runs.
    evaluation = {
        "regression_metrics": {
            "roc_auc":   {"value": metrics["roc_auc"]},
            "pr_auc":    {"value": metrics["pr_auc"]},
            "f1_score":  {"value": metrics["f1_score"]},
            "recall":    {"value": metrics["recall"]},
        }
    }
    eval_path = os.path.join(model_dir, "evaluation.json")
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"   evaluation.json saved (SageMaker-compatible format)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Local training pipeline for churn prediction — mirrors the SageMaker pipeline"
    )

    # Config file (defaults to the shared config.yaml one level up)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), '..', 'config.yaml'),
                        help="Path to config.yaml")

    # Data source override
    parser.add_argument("--source", choices=["s3", "redshift", "parquet"], default=None,
                        help="Data source. Overrides config.yaml. 's3' is fastest (default in config).")

    # Hyperparameter overrides
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--l2-leaf-reg", type=float, default=None)

    # Quality gate threshold overrides
    parser.add_argument("--roc-auc-threshold", type=float, default=0.7)
    parser.add_argument("--pr-auc-threshold",  type=float, default=0.5)
    parser.add_argument("--f1-threshold",       type=float, default=0.5)
    parser.add_argument("--recall-threshold",   type=float, default=0.7)

    # Output directory — defaults to models/<timestamp> so each run is kept separately
    parser.add_argument("--model-dir", default=None,
                        help="Directory to save the model (default: local_pipeline/models/<timestamp>/)")

    args = parser.parse_args()

    # Build timestamped model dir if not explicitly provided
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.model_dir is None:
        args.model_dir = os.path.join(os.path.dirname(__file__), 'models', run_timestamp)

    print("\n" + "="*60)
    print("LOCAL TRAINING PIPELINE — Churn Prediction")
    print("Mirrors the SageMaker pipeline but runs entirely on your machine")
    print("="*60)

    # Load config
    config = load_config(args.config)

    # Apply hyperparameter overrides
    if args.n_estimators:
        config['model']['n_estimators'] = args.n_estimators
    if args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate
    if args.depth:
        config['model']['depth'] = args.depth
    if args.l2_leaf_reg:
        config['model']['l2_leaf_reg'] = args.l2_leaf_reg

    # Quality gate thresholds
    thresholds = {
        'roc_auc':  args.roc_auc_threshold,
        'pr_auc':   args.pr_auc_threshold,
        'f1_score': args.f1_threshold,
        'recall':   args.recall_threshold,
    }

    print(f"\nConfiguration:")
    print(f"   Data source:   {args.source or config['data']['source']}")
    print(f"   n_estimators:  {config['model']['n_estimators']}")
    print(f"   learning_rate: {config['model']['learning_rate']}")
    print(f"   depth:         {config['model']['depth']}")
    print(f"   l2_leaf_reg:   {config['model']['l2_leaf_reg']}")
    print(f"   Thresholds:    ROC-AUC>={thresholds['roc_auc']}  PR-AUC>={thresholds['pr_auc']}  F1>={thresholds['f1_score']}  Recall>={thresholds['recall']}")
    print(f"   Model dir:     {args.model_dir}")

    start_time = datetime.now()

    # MLflow tracks every run automatically in a local mlruns/ folder.
    # To browse all runs: run `mlflow ui` in this directory, then open http://localhost:5000
    mlflow.set_experiment("churn-local-pipeline")
    with mlflow.start_run(run_name=f"run_{run_timestamp}"):

        # Log hyperparameters so you can compare runs side by side in the UI
        mlflow.log_params({
            "source":        args.source or config['data']['source'],
            "n_estimators":  config['model']['n_estimators'],
            "learning_rate": config['model']['learning_rate'],
            "depth":         config['model']['depth'],
            "l2_leaf_reg":   config['model']['l2_leaf_reg'],
        })

        # --- Run the 4 pipeline steps ---

        # Step 1: Load data
        df = step_load_data(config, source_override=args.source)

        # Log data provenance — where the data came from and how much
        source = args.source or config['data']['source']
        mlflow.log_params({
            "data_rows":    len(df),
            "data_columns": df.shape[1],
        })
        if source == 's3':
            s3_uri = config['data']['s3_uri']
            bucket = s3_uri.replace('s3://', '').split('/')[0]
            prefix = '/'.join(s3_uri.replace('s3://', '').split('/')[1:])
            mlflow.log_params({
                "data_s3_uri":    s3_uri,
                "data_s3_bucket": bucket,
                "data_s3_prefix": prefix,
            })
            # Log as MLflow dataset so it appears in the "Dataset" column of the UI
            dataset = mlflow.data.from_pandas(df, source=s3_uri, name="churn-training")
            mlflow.log_input(dataset, context="training")
        elif source == 'redshift':
            mlflow.log_params({
                "data_redshift_host":  config['data']['redshift_kwargs']['host'],
                "data_redshift_db":    config['data']['redshift_kwargs']['dbname'],
                "data_redshift_sql":   config['data']['redshift_sql'],
            })
            dataset = mlflow.data.from_pandas(df, source=config['data']['redshift_kwargs']['host'], name="churn-training-redshift")
            mlflow.log_input(dataset, context="training")
        elif source == 'parquet':
            mlflow.log_param("data_parquet_uri", config['data']['parquet_uri'])
            dataset = mlflow.data.from_pandas(df, source=config['data']['parquet_uri'], name="churn-training-parquet")
            mlflow.log_input(dataset, context="training")

        # Step 2: Train model
        model, X_train, X_valid, y_train, y_valid = step_train(df, config)

        # Step 3: Evaluate model
        metrics = step_evaluate(model, X_train, X_valid, y_train, y_valid, config)

        # Step 4: Quality gates — only save if all pass
        passed = step_quality_gates(metrics, thresholds)

        # Log all evaluation metrics to MLflow
        mlflow.log_metrics({
            "roc_auc":          metrics['roc_auc'],
            "pr_auc":           metrics['pr_auc'],
            "f1_score":         metrics['f1_score'],
            "recall":           metrics['recall'],
            "precision":        metrics['precision'],
            "accuracy":         metrics['accuracy'],
            # The threshold applied to probabilities to achieve the target recall
            "decision_threshold": metrics['threshold'],
        })

        # Tag whether the run passed quality gates — useful for filtering in the UI
        mlflow.set_tag("quality_gates_passed", str(passed))
        mlflow.set_tag("data_source", args.source or config['data']['source'])

        if passed:
            save_outputs(model, config, metrics, args.model_dir)
            # Log remaining artifacts (config, metrics, evaluation json)
            mlflow.log_artifacts(args.model_dir, artifact_path="artifacts")
            # Log the model using the CatBoost MLflow flavor — this populates the "Models" column
            mlflow.catboost.log_model(model, artifact_path="model")
        else:
            print("\nModel NOT saved (quality gates failed)")
            print("   Tip: lower the thresholds with --roc-auc-threshold, --recall-threshold, etc.")

        # Summary
        duration = datetime.now() - start_time
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"   Duration:  {duration.seconds}s")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print(f"   Recall:    {metrics['recall']:.1%}")
        print(f"   Precision: {metrics['precision']:.1%}")
        if passed:
            print(f"   Model saved to: {args.model_dir}")
        else:
            print(f"   Model NOT saved (quality gates failed)")
        print(f"   MLflow run logged — view with: mlflow ui")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
