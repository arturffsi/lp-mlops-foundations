"""
SageMaker training entry point for hyperparameter tuning.
This script is designed to work with SageMaker's hyperparameter tuning jobs.
"""

import argparse
import os
import sys
import subprocess
import yaml

def install_dependencies():
    """Install required dependencies if they're missing"""
    print("🔧 Installing dependencies...")
    
    # Check if requirements.txt exists in the current directory or code directory
    requirements_paths = ["requirements.txt", "/opt/ml/code/requirements.txt", "src/requirements.txt"]
    requirements_file = None
    
    for path in requirements_paths:
        if os.path.exists(path):
            requirements_file = path
            break
    
    if requirements_file:
        print(f"📦 Installing packages from {requirements_file}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Some dependencies may have failed to install: {e}")
    else:
        # Fallback: install just the critical dependencies
        critical_packages = ["catboost>=1.2.0", "pandas>=1.5.0", "scikit-learn>=1.2.0", "PyYAML>=6.0"]
        print("📦 Installing critical packages...")
        for package in critical_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to install {package}: {e}")

# Install dependencies first
install_dependencies()

from training_utils import (
    run_catboost_training, setup_mlflow_tracking, 
    preprocess_data, write_sagemaker_metrics, prepare_data_for_training
)
from data_io import load_data


def save_model_for_sagemaker(df, config, hyperparams, model_dir):
    """
    Train and save model specifically for SageMaker deployment.
    This creates the model.tar.gz artifact that appears in SageMaker Models.
    """
    import pickle
    import json
    from catboost import Pool
    
    # Retrain model with same parameters (needed since run_catboost_training only returns metrics)
    X_train, X_valid, y_train, y_valid, cat_idx = prepare_data_for_training(df, config)
    
    # Create Pools
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx)
    
    # Create and train model
    from training_utils import create_catboost_model
    model = create_catboost_model(config, hyperparams)
    
    print("🔄 Training model for SageMaker deployment...")
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=config['model']['early_stopping_rounds'],
        verbose=False  # Reduce output since we already trained once
    )
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names for inference
    feature_names_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(list(X_train.columns), f)
    
    # Save categorical feature indices
    cat_features_path = os.path.join(model_dir, "categorical_features.json")
    with open(cat_features_path, 'w') as f:
        json.dump(cat_idx, f)
    
    # Save config for reproducibility
    config_path = os.path.join(model_dir, "model_config.json")
    config_copy = config.copy()
    # Remove non-serializable items if any
    if hyperparams:
        config_copy['hyperparams_used'] = hyperparams
    with open(config_path, 'w') as f:
        json.dump(config_copy, f, indent=2, default=str)
    
    print(f"✅ Model saved to {model_path}")
    print(f"📄 Feature names saved to {feature_names_path}")
    print(f"🔢 Categorical features saved to {cat_features_path}")
    print(f"⚙️  Configuration saved to {config_path}")



def main():
    parser = argparse.ArgumentParser(description="SageMaker CatBoost churn model training")
    
    # Fixed configuration
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    # Hyperparameters for tuning - defaults are None so they don't override config unless specified
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--l2-leaf-reg", type=float, default=None)
    
    # SageMaker environment variables
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    
    # MLflow mode (consistent with train.py)
    parser.add_argument("--mlflow-mode", choices=['local', 'sagemaker', 'disabled'], default='local',
                       help="MLflow tracking mode: local (start local server), sagemaker (use SageMaker tracking server), or disabled")
    
    args = parser.parse_args()

    # Load configuration - check multiple possible locations
    config_paths = [
        args.config,  # Current directory
        f"/opt/ml/input/config/{args.config}",  # SageMaker input config
        f"/opt/ml/input/data/training/{args.config}",  # SageMaker training data
        "/opt/ml/code/config.yaml",  # In code directory
        "config.yaml"  # Default
    ]
    
    config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            break
    
    if config is None:
        raise FileNotFoundError(f"Config file not found in any of: {config_paths}")

    # Configure MLflow
    if args.mlflow_mode != 'disabled':
        setup_mlflow_tracking(config)

    # Load data
    data_source = config['data']['source']
    if data_source == 'parquet':
        data_uri = config['data']['parquet_uri']
        df = load_data(source=data_source, uri=data_uri)
    else:  # redshift
        df = load_data(
            source=data_source, 
            uri="",  # not used for redshift
            sql=config['data']['redshift_sql'],
            redshift_kwargs=config['data']['redshift_kwargs']
        )
    
    print(f"Loading data from {data_source}: {data_uri if data_source == 'parquet' else 'Redshift'}")
    print(f"Loaded data shape: {df.shape}")

    # Data preprocessing
    df = preprocess_data(df, config)

    # Print target distribution
    target_col = config['model']['target_col']
    if target_col in df.columns:
        print(f"\nTarget distribution:")
        print(df[target_col].value_counts(normalize=True))

    # Start with base hyperparameters from the config file
    hyperparams = config['model'].copy()

    # Create a dictionary of overrides from command-line arguments
    overrides = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'depth': args.depth,
        'l2_leaf_reg': args.l2_leaf_reg
    }

    # Filter out any arguments that were not explicitly provided (i.e., are None)
    provided_overrides = {k: v for k, v in overrides.items() if v is not None}

    # Update the base hyperparameters with any provided overrides
    if provided_overrides:
        print(f"\nOverriding config with provided hyperparameters: {provided_overrides}")
        hyperparams.update(provided_overrides)

    print(f"\nFinal Hyperparameters: {hyperparams}")

    # Train model using shared function
    if args.mlflow_mode == 'disabled':
        # Run training without MLflow context
        metrics = run_catboost_training(
            df, config, hyperparams=hyperparams, 
            data_source=data_source, data_uri=data_uri,
            mlflow_enabled=False
        )
    else:
        # Run training with MLflow context
        import mlflow
        with mlflow.start_run():
            metrics = run_catboost_training(
                df, config, hyperparams=hyperparams, 
                data_source=data_source, data_uri=data_uri,
                mlflow_enabled=True
            )
    
    print(f"\nTraining completed successfully!")
    print(f"Final ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Final F1 @ target recall: {metrics['f1_score']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Churner recall: {metrics['churner_recall']:.4f}")
    print(f"Churner precision: {metrics['churner_precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Total churners: {metrics['total_churners']} ({metrics['actual_churn_rate']:.2f}%)")
    print(f"Train samples: {metrics['train_samples']}")

    # Write metrics for SageMaker hyperparameter tuning
    write_sagemaker_metrics(metrics)
    
    # Save model to SageMaker model directory for deployment
    print(f"\n💾 Saving model to SageMaker model directory: {args.model_dir}")
    save_model_for_sagemaker(df, config, hyperparams, args.model_dir)
    
    # Note: Model registration happens after training completes
    print(f"\n💡 Model artifacts will be uploaded to S3 after training completes")
    print(f"🎯 To create a SageMaker Model, run after job completes:")
    print(f"   python src/create_model_from_job.py --training-job-name [JOB_NAME]")


if __name__ == "__main__":
    main()