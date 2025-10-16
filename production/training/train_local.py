import argparse
import yaml
import mlflow
from training_utils import (
    run_catboost_training, setup_mlflow_tracking, 
    preprocess_data
)
from data_io import load_data



def main():
    parser = argparse.ArgumentParser(description="Train CatBoost churn model")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--mlflow-mode", choices=['sagemaker', 'disabled'], default='sagemaker', 
                       help="MLflow tracking mode: sagemaker (use SageMaker tracking server), or disabled")
    
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Configure MLflow if not disabled
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

    # Train model using shared function
    if args.mlflow_mode == 'disabled':
        # Run training without MLflow context
        metrics = run_catboost_training(
            df, config, data_source=data_source, data_uri=data_uri,
            mlflow_enabled=False
        )
    else:
        # Run training with MLflow context
        with mlflow.start_run():
            metrics = run_catboost_training(
                df, config, data_source=data_source, data_uri=data_uri,
                mlflow_enabled=True
            )
    
    print(f"\nTraining completed successfully!")
    print(f"Final ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Final F1 @ target recall: {metrics['f1_score']:.4f}")
    print(f"Final threshold: {metrics['threshold']:.4f}")
    
    if args.mlflow_mode != 'disabled' and 'registered_model_uri' in metrics:
        print(f"\n📍 Model saved in MLflow: {metrics['registered_model_uri']}")
    else:
        print(f"\n✅ Model training completed (no registration)")


if __name__ == "__main__":
    main()