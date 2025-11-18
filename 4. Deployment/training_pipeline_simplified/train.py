"""
Simplified SageMaker Training Script for Churn Prediction

This is a beginner-friendly version that still runs on SageMaker.
Clean code, clear steps, easy to understand.

Usage:
  python train.py                    # Local training
  python train.py --mlflow-mode disabled  # Disable MLflow
"""

import argparse
import os
import yaml

# Import our simplified utilities
from data_loader import load_data
from train_utils import (
    preprocess_data,
    prepare_features,
    train_model,
    evaluate_model,
    save_model,
    write_sagemaker_metrics
)


def load_config(config_path):
    """Load configuration from YAML file."""
    # Try multiple locations (for SageMaker compatibility)
    possible_paths = [
        config_path,
        f"/opt/ml/input/config/{config_path}",
        f"/opt/ml/code/{config_path}",
        "config.yaml"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"📋 Loading config from: {path}")
            with open(path, 'r') as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(f"Config not found in: {possible_paths}")


def main():
    """Main training pipeline."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simplified SageMaker churn model training")

    # Configuration
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    # Hyperparameters (can override config values)
    parser.add_argument("--n-estimators", type=int, help="Number of trees")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--depth", type=int, help="Tree depth")
    parser.add_argument("--l2-leaf-reg", type=float, help="L2 regularization")

    # SageMaker paths
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    # MLflow mode
    parser.add_argument("--mlflow-mode", choices=['sagemaker', 'disabled'],
                       default='disabled', help="MLflow tracking mode")

    args = parser.parse_args()

    # Print header
    print("\n" + "="*60)
    print("🎯 CHURN PREDICTION MODEL TRAINING")
    print("   Simplified SageMaker Version")
    print("="*60 + "\n")

    # 1. Load configuration
    config = load_config(args.config)

    # Override hyperparameters from command line
    if args.n_estimators:
        config['model']['n_estimators'] = args.n_estimators
    if args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate
    if args.depth:
        config['model']['depth'] = args.depth
    if args.l2_leaf_reg:
        config['model']['l2_leaf_reg'] = args.l2_leaf_reg

    print(f"⚙️  Hyperparameters:")
    print(f"   n_estimators: {config['model']['n_estimators']}")
    print(f"   learning_rate: {config['model']['learning_rate']}")
    print(f"   depth: {config['model']['depth']}")
    print(f"   l2_leaf_reg: {config['model']['l2_leaf_reg']}")
    print()

    # Setup MLflow if enabled
    if args.mlflow_mode == 'sagemaker':
        import mlflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        mlflow.start_run()
        print("✅ MLflow tracking enabled\n")

    # 2. Load data
    df = load_data(config)

    # 3. Preprocess data
    df = preprocess_data(df, config)

    # 4. Prepare features and split data
    X_train, X_valid, y_train, y_valid, cat_indices = prepare_features(df, config)

    # 5. Train model
    model = train_model(X_train, X_valid, y_train, y_valid, cat_indices, config)

    # 6. Evaluate model
    metrics = evaluate_model(model, X_valid, y_valid, config, train_size=len(y_train))

    # 7. Log to MLflow if enabled
    if args.mlflow_mode == 'sagemaker':
        import mlflow
        mlflow.log_params({
            'n_estimators': config['model']['n_estimators'],
            'learning_rate': config['model']['learning_rate'],
            'depth': config['model']['depth'],
            'l2_leaf_reg': config['model']['l2_leaf_reg']
        })
        mlflow.log_metrics({
            'roc_auc': metrics['roc_auc'],
            'f1_score': metrics['f1_score'],
            'recall': metrics['recall'],
            'precision': metrics['precision']
        })
        mlflow.catboost.log_model(model, "model")
        mlflow.end_run()

    # 8. Save model for SageMaker
    save_model(model, config, args.model_dir)

    # 9. Write metrics for SageMaker
    write_sagemaker_metrics(metrics)

    # Print summary
    print("="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"🎯 Final ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"🎯 Final F1 Score: {metrics['f1_score']:.4f}")
    print(f"🎯 Recall: {metrics['recall']:.1%}")
    print(f"🎯 Precision: {metrics['precision']:.1%}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
