"""
SageMaker training entry point optimized for large datasets (millions of rows).
This script implements memory-efficient data loading and processing for massive datasets.
"""

import argparse
import os
import yaml
import gc
import psutil
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Iterator
from training_utils import (
    setup_mlflow_tracking, write_sagemaker_metrics, preprocess_data,
    create_catboost_model, target_recall_threshold, parse_maybe_yyyymmdd
)
from data_io import load_data


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(stage: str):
    """Log memory usage at different stages."""
    memory_mb = get_memory_usage()
    print(f"💾 Memory usage at {stage}: {memory_mb:.1f} MB")


def load_data_with_sampling(source: str, uri: str, sample_ratio: float = None, 
                           random_seed: int = 42, sql: str = None, 
                           redshift_kwargs: dict = None) -> pd.DataFrame:
    """
    Load data with optional sampling for memory efficiency.
    
    Args:
        source: 'parquet' or 'redshift'
        uri: Data location
        sample_ratio: If provided, randomly sample this fraction of data (0.0-1.0)
        random_seed: Random seed for sampling
        sql: SQL query for redshift
        redshift_kwargs: Redshift connection parameters
    """
    print(f"🔄 Loading data from {source}: {uri}")
    log_memory_usage("before data loading")
    
    if source == "parquet":
        # For large parquet files, we can use sampling
        df = pd.read_parquet(uri, engine="pyarrow")
        
        if sample_ratio and sample_ratio < 1.0:
            original_size = len(df)
            df = df.sample(frac=sample_ratio, random_state=random_seed).reset_index(drop=True)
            print(f"📊 Sampled {len(df):,} rows from {original_size:,} ({sample_ratio:.1%})")
    
    elif source == "redshift":
        # For Redshift, add TABLESAMPLE to SQL if sampling is requested
        if sample_ratio and sample_ratio < 1.0:
            # Add TABLESAMPLE SYSTEM to the SQL query if not already present
            if "TABLESAMPLE" not in sql.upper():
                # Find the FROM clause and add sampling
                from_pos = sql.upper().find("FROM")
                if from_pos != -1:
                    # Find the next space or WHERE after table name
                    remaining = sql[from_pos:]
                    table_end = remaining.find("WHERE")
                    if table_end == -1:
                        table_end = remaining.find("ORDER")
                    if table_end == -1:
                        table_end = remaining.find("GROUP")
                    if table_end == -1:
                        table_end = remaining.find("LIMIT")
                    if table_end == -1:
                        table_end = len(remaining)
                    
                    # Insert TABLESAMPLE after table name
                    before = sql[:from_pos + table_end]
                    after = sql[from_pos + table_end:]
                    sample_percent = max(1, int(sample_ratio * 100))  # At least 1%
                    sql = f"{before} TABLESAMPLE SYSTEM ({sample_percent}) {after}"
                    print(f"📊 Added TABLESAMPLE SYSTEM ({sample_percent}%) to SQL")
        
        from redshift_connector import connect
        conn = connect(**redshift_kwargs)
        try:
            df = pd.read_sql(sql, conn)
        finally:
            conn.close()
    
    else:
        raise ValueError(f"Unknown source: {source}")
    
    log_memory_usage("after data loading")
    print(f"📊 Loaded data shape: {df.shape}")
    return df


def chunk_data_processing(df: pd.DataFrame, config: dict, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Process data in chunks to reduce memory usage.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        chunk_size: Number of rows to process at once
    """
    if len(df) <= chunk_size:
        print("📊 Dataset fits in memory, processing normally")
        return preprocess_data(df, config)
    
    print(f"🔄 Processing data in chunks of {chunk_size:,} rows")
    log_memory_usage("before chunk processing")
    
    chunks = []
    total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        print(f"  Processing chunk {i+1}/{total_chunks} (rows {start_idx:,} to {end_idx-1:,})")
        
        # Process chunk
        chunk = df.iloc[start_idx:end_idx].copy()
        processed_chunk = preprocess_data(chunk, config)
        chunks.append(processed_chunk)
        
        # Force garbage collection
        del chunk
        gc.collect()
        
        if (i + 1) % 5 == 0:  # Log memory every 5 chunks
            log_memory_usage(f"after chunk {i+1}")
    
    # Concatenate all processed chunks
    print("🔄 Concatenating processed chunks...")
    result = pd.concat(chunks, ignore_index=True)
    
    # Clean up
    del chunks
    gc.collect()
    
    log_memory_usage("after chunk processing complete")
    print(f"✅ Chunk processing complete. Final shape: {result.shape}")
    
    return result


def memory_efficient_train_test_split(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list]:
    """
    Memory-efficient train/test split that processes data in place where possible.
    """
    TARGET_COL = config['model']['target_col']
    id_features = config['features']['id_features']
    
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found."
    
    log_memory_usage("before train/test split")
    
    # Drop ID features in place
    df.drop(columns=[col for col in id_features if col in df.columns], inplace=True)
    
    # Extract target
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL], inplace=False)
    
    # Time-based split if possible
    date_col = next((c for c in config['features']['datetime_features'] 
                    if c in df.columns and 'date_inicio' in c), None)
    
    if date_col is not None:
        dates = parse_maybe_yyyymmdd(df[date_col])
        if dates.notna().mean() > 0.8:
            cutoff = dates.quantile(0.8)
            print(f'📅 Time-based split with cutoff_date: {cutoff}')
            train_mask = dates < cutoff
            valid_mask = ~train_mask
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=config['training']['test_size'], 
                random_state=config['model']['random_seed']
            )
            train_idx, valid_idx = next(splitter.split(X, y))
            train_mask = pd.Series(False, index=X.index)
            train_mask.iloc[train_idx] = True
            valid_mask = ~train_mask
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=config['training']['test_size'], 
            random_state=config['model']['random_seed']
        )
        train_idx, valid_idx = next(splitter.split(X, y))
        train_mask = pd.Series(False, index=X.index)
        train_mask.iloc[train_idx] = True
        valid_mask = ~train_mask
    
    # Drop datetime columns (CatBoost can't handle NaT)
    dt_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    if dt_cols:
        X.drop(columns=dt_cols, inplace=True)
    
    # Handle categorical features efficiently
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    cat_idx = X.columns.get_indexer(cat_cols).tolist()
    
    # Fix categorical NaNs for CatBoost - process in place
    for c in cat_cols:
        s = X[c]
        if str(s.dtype) == "category":
            if '<MISSING>' not in s.cat.categories:
                s = s.cat.add_categories(['<MISSING>'])
            X[c] = s.fillna('<MISSING>').astype(object)
        else:
            X[c] = s.astype(object).where(~pd.isna(s), '<MISSING>')
    
    # Create splits - use views where possible to save memory
    X_train = X.loc[train_mask].copy()
    X_valid = X.loc[valid_mask].copy()
    y_train = y[train_mask]
    y_valid = y[valid_mask]
    
    # Force garbage collection
    del X, df
    gc.collect()
    
    log_memory_usage("after train/test split")
    
    print(f"📊 Train samples: {len(X_train):,}, Valid samples: {len(X_valid):,}")
    print(f"📊 Train churn rate: {y_train.mean():.3f}, Valid churn rate: {y_valid.mean():.3f}")
    
    return X_train, X_valid, y_train, y_valid, cat_idx


def train_catboost_large_dataset(X_train: pd.DataFrame, X_valid: pd.DataFrame, 
                                y_train: np.ndarray, y_valid: np.ndarray, 
                                cat_idx: list, config: dict, 
                                hyperparams: dict = None) -> Tuple[dict, pd.DataFrame]:
    """
    Train CatBoost model with memory-efficient settings for large datasets.
    """
    from catboost import Pool
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
    
    log_memory_usage("before model training")
    
    # Create model with memory-efficient settings for large datasets
    model = create_catboost_model(config, hyperparams)
    
    # For very large datasets, we can use border_count and other memory optimizations
    if len(X_train) > 1000000:  # 1M+ rows
        print("🔧 Applying memory optimizations for large dataset")
        # Reduce border_count for memory efficiency
        model.set_params(border_count=32)  # Default is 254, reducing saves memory
        model.set_params(max_ctr_complexity=1)  # Reduce categorical feature complexity
    
    # Create Pools - CatBoost will handle large data efficiently
    print("🔄 Creating CatBoost Pools...")
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx)
    
    log_memory_usage("after creating pools")
    
    # Train model
    print("🚀 Starting model training...")
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=config['model']['early_stopping_rounds'],
        verbose=500  # Less frequent logging for large datasets
    )
    
    log_memory_usage("after model training")
    
    # Efficient prediction - process in batches if needed
    print("🔄 Generating predictions...")
    
    if len(X_valid) > 500000:  # 500k+ rows
        print("📊 Using batched prediction for large validation set")
        batch_size = 100000
        valid_proba = []
        
        for i in range(0, len(X_valid), batch_size):
            batch_end = min(i + batch_size, len(X_valid))
            batch_pool = Pool(X_valid.iloc[i:batch_end], cat_features=cat_idx)
            batch_proba = model.predict_proba(batch_pool)[:, 1]
            valid_proba.extend(batch_proba)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Processed {batch_end:,} / {len(X_valid):,} predictions")
        
        valid_proba = np.array(valid_proba)
    else:
        valid_proba = model.predict_proba(valid_pool)[:, 1]
    
    # Calculate metrics
    roc = roc_auc_score(y_valid, valid_proba)
    pr_auc = average_precision_score(y_valid, valid_proba)
    
    # Target recall threshold
    target_recall = config['training']['target_recall_threshold']
    thr, r_at_thr, p_at_thr = target_recall_threshold(y_valid, valid_proba, target_recall)
    y_pred = (valid_proba >= thr).astype(int)
    
    # Calculate F1 at target recall threshold
    f1_at_threshold = 2 * p_at_thr * r_at_thr / (p_at_thr + r_at_thr) if (p_at_thr + r_at_thr) > 0 else 0
    
    # Confusion matrix and derived metrics
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    accuracy = accuracy_score(y_valid, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Business metrics
    total_churners = int(tp + fn)
    actual_churn_rate = total_churners / len(y_valid) if len(y_valid) > 0 else 0.0
    predicted_churners = int(tp + fp)
    predicted_churn_rate = predicted_churners / len(y_valid) if len(y_valid) > 0 else 0.0
    
    # Print metrics
    print("\n📊 Validation Metrics (Large Dataset)")
    print("=" * 40)
    print(f"ROC-AUC:                {roc:.4f}")
    print(f"PR-AUC:                 {pr_auc:.4f}")
    print(f"Target recall:          {target_recall:.1%}")
    print(f"Threshold:              {thr:.4f}")
    print(f"F1 @ target recall:     {f1_at_threshold:.4f}")
    print(f"Churner recall:         {r_at_thr:.4f} ({r_at_thr:.1%})")
    print(f"Churner precision:      {p_at_thr:.4f} ({p_at_thr:.1%})")
    print(f"Accuracy:               {accuracy:.4f} ({accuracy:.1%})")
    print(f"Total churners:         {total_churners:,} ({actual_churn_rate:.1%})")
    print(f"Predicted churners:     {predicted_churners:,} ({predicted_churn_rate:.1%})")
    
    # Feature importance (memory-efficient)
    print("🔄 Computing feature importance...")
    feature_importance = model.get_feature_importance(train_pool, type="FeatureImportance")
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)
    
    print("\n🏆 Top 20 Features:")
    print(fi.head(20).to_string(index=False))
    
    log_memory_usage("after evaluation")
    
    # Comprehensive metrics dictionary
    metrics = {
        # Primary metrics
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "f1_score": f1_at_threshold,
        "threshold": thr,
        "precision": p_at_thr,
        "recall": r_at_thr,
        
        # Churner-specific metrics
        "churner_recall": r_at_thr,
        "churner_precision": p_at_thr,
        
        # Business metrics
        "actual_churn_rate": actual_churn_rate,
        "predicted_churn_rate": predicted_churn_rate,
        "total_churners": total_churners,
        "predicted_churners": predicted_churners,
        
        # Performance metrics
        "accuracy": accuracy,
        "specificity": specificity,
        
        # Sample counts
        "train_samples": len(X_train),
        "valid_samples": len(X_valid),
        "feature_count": len(X_train.columns),
        
        # Confusion matrix
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }
    
    return metrics, fi, model



def save_model_for_sagemaker_large(model, X_train: pd.DataFrame, config: dict, 
                                  hyperparams: dict, model_dir: str, cat_idx: list):
    """
    Save model artifacts optimized for large dataset scenarios.
    """
    import pickle
    import json
    
    print("💾 Saving model for SageMaker deployment...")
    log_memory_usage("before model saving")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model (CatBoost models are already optimized)
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names (essential for inference)
    feature_names_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(list(X_train.columns), f)
    
    # Save categorical feature indices
    cat_features_path = os.path.join(model_dir, "categorical_features.json")
    with open(cat_features_path, 'w') as f:
        json.dump(cat_idx, f)
    
    # Save config with hyperparameters
    config_path = os.path.join(model_dir, "model_config.json")
    config_copy = config.copy()
    if hyperparams:
        config_copy['hyperparams_used'] = hyperparams
    config_copy['model_type'] = 'large_dataset_optimized'
    config_copy['training_samples'] = len(X_train)
    
    with open(config_path, 'w') as f:
        json.dump(config_copy, f, indent=2, default=str)
    
    # Save memory usage info
    memory_info_path = os.path.join(model_dir, "training_memory_info.json")
    with open(memory_info_path, 'w') as f:
        json.dump({
            "peak_memory_mb": get_memory_usage(),
            "training_optimizations": "large_dataset_mode",
            "catboost_optimizations": ["reduced_border_count", "limited_ctr_complexity"]
        }, f, indent=2)
    
    log_memory_usage("after model saving")
    
    print(f"✅ Model saved to {model_path}")
    print(f"📄 Feature names saved to {feature_names_path}")
    print(f"🔢 Categorical features saved to {cat_features_path}")
    print(f"⚙️  Configuration saved to {config_path}")
    print(f"💾 Memory info saved to {memory_info_path}")


def main():
    parser = argparse.ArgumentParser(description="SageMaker CatBoost churn model training optimized for large datasets")
    
    # Configuration
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    
    # Hyperparameters for tuning - defaults are None so they don't override config unless specified
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--l2-leaf-reg", type=float, default=None)
    
    # Memory optimization parameters
    parser.add_argument("--sample-ratio", type=float, default=None,
                       help="Sample this fraction of data for memory efficiency (0.0-1.0)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Exact number of rows to sample (e.g., 5000000 for 5M rows)")
    parser.add_argument("--chunk-size", type=int, default=100000,
                       help="Chunk size for data processing")
    parser.add_argument("--memory-limit-gb", type=float, default=None,
                       help="Memory limit in GB (will enable aggressive optimizations)")
    
    # SageMaker environment
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    
    # MLflow mode
    parser.add_argument("--mlflow-mode", choices=['local', 'sagemaker', 'disabled'], 
                       default='disabled', help="MLflow tracking mode")
    
    args = parser.parse_args()

    # Handle sample-size vs sample-ratio conflict
    if args.sample_size and args.sample_ratio:
        raise ValueError("Cannot specify both --sample-size and --sample-ratio. Please use one or the other.")

    # Print system info
    print("🖥️  System Information")
    print("=" * 30)
    print(f"Available CPU cores: {os.cpu_count()}")
    print(f"Available memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if args.memory_limit_gb:
        print(f"Memory limit set: {args.memory_limit_gb} GB")
    if args.sample_ratio:
        print(f"Data sampling: {args.sample_ratio:.1%}")
    elif args.sample_size:
        print(f"Data sampling: {args.sample_size:,} rows (exact)")
    print(f"Chunk size: {args.chunk_size:,} rows")
    
    log_memory_usage("script start")
    
    # Load configuration
    config_paths = [
        args.config,
        f"/opt/ml/input/config/{args.config}",
        f"/opt/ml/input/data/training/{args.config}",
        "/opt/ml/code/config.yaml",
        "config.yaml"
    ]
    
    config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"📖 Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            break
    
    if config is None:
        raise FileNotFoundError(f"Config file not found in any of: {config_paths}")
    
    # Configure MLflow if needed
    if args.mlflow_mode != 'disabled':
        setup_mlflow_tracking(config)
        import mlflow
        mlflow.start_run()
    
    # Load data with optional sampling
    data_source = config['data']['source']
    if data_source == 'parquet':
        # Check if we're in SageMaker environment first
        sagemaker_data_dir = "/opt/ml/input/data/training"
        if os.path.exists(sagemaker_data_dir):
            # Use SageMaker mounted data directory - load directly with pandas
            print(f"🔄 Running in SageMaker - loading from mounted directory: {sagemaker_data_dir}")
            import glob
            parquet_files = glob.glob(f"{sagemaker_data_dir}/*.parquet")
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {sagemaker_data_dir}")

            print(f"📂 Found {len(parquet_files)} parquet files")
            dfs = []
            for file in parquet_files:
                print(f"📄 Loading: {os.path.basename(file)}")
                dfs.append(pd.read_parquet(file, engine="pyarrow"))

            df = pd.concat(dfs, ignore_index=True)
            print(f"📊 Loaded {len(df):,} rows from SageMaker data")

            # Apply sampling if requested
            original_size = len(df)
            if args.sample_size:
                # Sample exact number of rows
                if args.sample_size < original_size:
                    df = df.sample(n=args.sample_size, random_state=config['model']['random_seed'])
                    sample_pct = (args.sample_size / original_size) * 100
                    print(f"🔀 Sampled {len(df):,} rows ({sample_pct:.1f}%) from {original_size:,}")
                else:
                    print(f"⚠️  Requested {args.sample_size:,} rows but dataset only has {original_size:,} - using full dataset")
            elif args.sample_ratio and args.sample_ratio < 1.0:
                # Sample by ratio
                df = df.sample(frac=args.sample_ratio, random_state=config['model']['random_seed'])
                print(f"🔀 Sampled {len(df):,} rows ({args.sample_ratio:.1%}) from {original_size:,}")
        else:
            # Use config file URI for local runs
            data_uri = config['data']['parquet_uri']
            print(f"🔄 Running locally - using config URI: {data_uri}")
            if args.sample_size:
                # Load data and then sample exact number of rows
                df = load_data_with_sampling(
                    source=data_source,
                    uri=data_uri,
                    sample_ratio=None,  # Don't sample in load_data_with_sampling
                    random_seed=config['model']['random_seed']
                )
                original_size = len(df)
                if args.sample_size < original_size:
                    df = df.sample(n=args.sample_size, random_state=config['model']['random_seed'])
                    sample_pct = (args.sample_size / original_size) * 100
                    print(f"🔀 Sampled {len(df):,} rows ({sample_pct:.1f}%) from {original_size:,}")
                else:
                    print(f"⚠️  Requested {args.sample_size:,} rows but dataset only has {original_size:,} - using full dataset")
            else:
                # Use ratio sampling
                df = load_data_with_sampling(
                    source=data_source,
                    uri=data_uri,
                    sample_ratio=args.sample_ratio,
                    random_seed=config['model']['random_seed']
                )
    else:  # redshift
        df = load_data_with_sampling(
            source=data_source,
            uri="",
            sql=config['data']['redshift_sql'],
            redshift_kwargs=config['data']['redshift_kwargs'],
            sample_ratio=args.sample_ratio,
            random_seed=config['model']['random_seed']
        )
    
    print(f"📊 Final loaded data shape: {df.shape}")
    
    # Memory-efficient data preprocessing
    df = chunk_data_processing(df, config, args.chunk_size)
    
    # Print target distribution
    target_col = config['model']['target_col']
    if target_col in df.columns:
        churn_rate = df[target_col].mean()
        print(f"\n📊 Target distribution:")
        print(f"  Churn rate: {churn_rate:.1%}")
        print(f"  Total samples: {len(df):,}")
        print(f"  Churners: {int(df[target_col].sum()):,}")
        print(f"  Non-churners: {int((~df[target_col].astype(bool)).sum()):,}")
    
    # Memory-efficient train/test split
    X_train, X_valid, y_train, y_valid, cat_idx = memory_efficient_train_test_split(df, config)
    
    # Clean up original dataframe
    del df
    gc.collect()
    log_memory_usage("after data preprocessing")
    
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

    print(f"\n🎛️  Final Hyperparameters Used:")
    print("=" * 50)
    for param, value in hyperparams.items():
        if param not in ['target_col']:
             print(f"  {param:<25}: {value}")
    print("=" * 50)
    
    # Train model
    print("\n🚀 Starting Large Dataset Training Pipeline")
    print("=" * 50)
    
    metrics, fi, model = train_catboost_large_dataset(
        X_train, X_valid, y_train, y_valid, cat_idx, config, hyperparams
    )
    
    # Log to MLflow if enabled
    if args.mlflow_mode != 'disabled':
        import mlflow
        mlflow.log_params(hyperparams)
        mlflow.log_metrics({
            'roc_auc': metrics['roc_auc'],
            'f1_score': metrics['f1_score'],
            'churner_recall': metrics['churner_recall'],
            'churner_precision': metrics['churner_precision'],
            'train_samples': metrics['train_samples'],
            'valid_samples': metrics['valid_samples']
        })
        mlflow.end_run()
    
    # Write SageMaker metrics
    write_sagemaker_metrics(metrics)
    
    # Save model for SageMaker deployment
    save_model_for_sagemaker_large(model, X_train, config, hyperparams, args.model_dir, cat_idx)
    
    # Final memory usage
    log_memory_usage("script end")
    
    print(f"\n✅ Large Dataset Training Complete!")
    print(f"📊 Final ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"📊 Final F1 @ target recall: {metrics['f1_score']:.4f}")
    print(f"💾 Peak memory usage: {get_memory_usage():.1f} MB")
    
    print(f"\n💡 Model artifacts saved to: {args.model_dir}")
    print(f"🎯 To create SageMaker Model after job completes:")
    print(f"   python src/create_model_from_job.py --training-job-name [JOB_NAME]")


if __name__ == "__main__":
    main()