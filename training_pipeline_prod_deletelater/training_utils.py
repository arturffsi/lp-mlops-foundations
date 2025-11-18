"""
Shared training utilities for CatBoost churn modeling.
This module contains common functions used by both train.py and train_sagemaker.py
"""

import numpy as np
import pandas as pd
import os
import hashlib
import json
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, precision_score, recall_score
)
from sklearn.model_selection import StratifiedShuffleSplit
# MLflow imports are conditional - only imported when needed


def target_recall_threshold(y_true, y_scores, target_recall=0.8):
    """Find threshold that achieves target recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find highest threshold that still achieves target recall
    idx = np.where(recalls[:-1] >= target_recall)[0]
    if len(idx):
        i = idx[-1]
        return thresholds[i], recalls[i], precisions[i]
    else:
        return 0.0, 0.0, 0.0


def parse_maybe_yyyymmdd(s):
    """Robustly parse YYYYMMDD ints/strings or general datelike values."""
    if pd.api.types.is_integer_dtype(s) or (s.dtype == object and s.astype(str).str.fullmatch(r"\d{8}").all()):
        return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def create_dataset_metadata(df, data_source, data_uri):
    """Create comprehensive dataset metadata for MLflow logging."""
    # Calculate data hash for versioning
    data_string = df.to_string(index=False)
    data_hash = hashlib.md5(data_string.encode()).hexdigest()[:12]
    
    # Generate dataset statistics
    metadata = {
        "source": data_source,
        "uri": data_uri,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_hash": data_hash,
        "created_at": datetime.now().isoformat(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }
    
    # Add target distribution if available
    if 'churn' in df.columns:
        metadata["target_distribution"] = df['churn'].value_counts().to_dict()
        metadata["churn_rate"] = float(df['churn'].mean())
    
    return metadata, data_hash


def log_dataset_to_mlflow(df, data_source, data_uri, context="training"):
    """Log dataset to MLflow with comprehensive metadata."""
    import mlflow
    import mlflow.data
    try:
        # Create dataset metadata
        metadata, data_hash = create_dataset_metadata(df, data_source, data_uri)
        
        # Create MLflow dataset
        dataset = mlflow.data.from_pandas(
            df,
            source=data_uri,
            name=f"churn_dataset_{data_hash}",
            digest=data_hash
        )
        
        # Log the dataset
        mlflow.log_input(dataset, context=context)
        
        # Log dataset metadata as parameters and metrics
        mlflow.log_params({
            f"data_{context}_source": data_source,
            f"data_{context}_uri": data_uri,
            f"data_{context}_hash": data_hash,
            f"data_{context}_shape": f"{metadata['shape'][0]}x{metadata['shape'][1]}",
            f"data_{context}_columns": len(metadata['columns']),
            f"data_{context}_created": metadata['created_at']
        })
        
        mlflow.log_metrics({
            f"data_{context}_rows": metadata['shape'][0],
            f"data_{context}_cols": metadata['shape'][1],
            f"data_{context}_memory_mb": metadata['memory_usage_mb'],
            f"data_{context}_missing_total": sum(metadata['missing_values'].values())
        })
        
        if 'churn_rate' in metadata:
            mlflow.log_metric(f"data_{context}_churn_rate", metadata['churn_rate'])
        
        # Save detailed metadata as artifact
        metadata_path = f"dataset_metadata_{context}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        mlflow.log_artifact(metadata_path, "datasets")
        
        # Clean up temp file
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            
        print(f"✅ Dataset logged to MLflow (context: {context})")
        print(f"📊 Dataset hash: {data_hash}")
        print(f"📈 Shape: {metadata['shape'][0]:,} rows × {metadata['shape'][1]} columns")
        print(f"💾 Memory: {metadata['memory_usage_mb']} MB")
        
        return dataset, data_hash
        
    except Exception as e:
        print(f"⚠️ Failed to log dataset to MLflow: {e}")
        return None, None


def prepare_data_for_training(df, config):
    """Prepare data for CatBoost training with proper preprocessing."""
    TARGET_COL = config['model']['target_col']
    id_features = config['features']['id_features']
    
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found."

    # Drop ID features
    df = df.drop(columns=[col for col in id_features if col in df.columns])

    # Target and features
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    # Time-based split if possible
    date_col = next((c for c in config['features']['datetime_features'] if c in df.columns and 'date_inicio' in c), None)
    
    if date_col is not None:
        dates = parse_maybe_yyyymmdd(df[date_col])
        if dates.notna().mean() > 0.8:
            cutoff = dates.quantile(0.8)
            print(f'Time-based split with cutoff_date: {cutoff}')
            train_mask = dates < cutoff
            valid_mask = ~train_mask
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=config['training']['test_size'], random_state=config['model']['random_seed']
            )
            train_idx, valid_idx = next(splitter.split(X, y))
            train_mask = pd.Series(False, index=X.index)
            train_mask.iloc[train_idx] = True
            valid_mask = ~train_mask
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=config['training']['test_size'], random_state=config['model']['random_seed']
        )
        train_idx, valid_idx = next(splitter.split(X, y))
        train_mask = pd.Series(False, index=X.index)
        train_mask.iloc[train_idx] = True
        valid_mask = ~train_mask

    # Drop datetime columns (CatBoost can't handle NaT)
    dt_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    if dt_cols:
        X = X.drop(columns=dt_cols).copy()

    # Handle categorical features (exclude numerical features)
    numerical_features = set()
    if 'int_features' in config['features']:
        numerical_features.update(config['features']['int_features'])
    if 'float_features' in config['features']:
        numerical_features.update(config['features']['float_features'])
    if 'datetime_features' in config['features']:
        numerical_features.update(config['features']['datetime_features'])

    cat_cols = [c for c in X.columns
                if (X[c].dtype == "object" or str(X[c].dtype) == "category")
                and c not in numerical_features]
    cat_idx = X.columns.get_indexer(cat_cols).tolist()

    # Fix categorical NaNs for CatBoost
    for c in cat_cols:
        s = X[c]
        if str(s.dtype) == "category":
            if '<MISSING>' not in s.cat.categories:
                s = s.cat.add_categories(['<MISSING>'])
            s = s.fillna('<MISSING>').astype(object)
        else:
            s = s.astype(object)
            s = s.where(~pd.isna(s), '<MISSING>')
        X[c] = s

    # Split data
    X_train, y_train = X.loc[train_mask], y[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y[valid_mask]

    print(f"Train samples: {len(X_train)}, Valid samples: {len(X_valid)}")
    print(f"Train churn rate: {y_train.mean():.3f}, Valid churn rate: {y_valid.mean():.3f}")

    return X_train, X_valid, y_train, y_valid, cat_idx


def create_catboost_model(config, hyperparams=None):
    """Create CatBoost model with given configuration and hyperparameters."""
    # Start with all parameters from the config file
    model_params = config['model'].copy()

    # Rename config keys to match CatBoost parameter names
    if 'n_estimators' in model_params:
        model_params['iterations'] = model_params.pop('n_estimators')
    if 'use_gpu' in model_params:
        model_params['task_type'] = "GPU" if model_params.pop('use_gpu') else "CPU"

    # If tuning hyperparameters are provided, they override the config
    if hyperparams:
        # Also rename keys for provided hyperparams
        if 'n_estimators' in hyperparams:
            hyperparams['iterations'] = hyperparams.pop('n_estimators')
        model_params.update(hyperparams)

    # Remove keys that are not valid CatBoost parameters before passing to the classifier
    # 'target_col' and 'early_stopping_rounds' are for the training process, not the model itself.
    invalid_keys = ['target_col', 'early_stopping_rounds', 'custom_metrics', 'use_gpu']
    for key in invalid_keys:
        if key in model_params:
            del model_params[key]
            
    # Add a default verbose setting if not specified
    if 'verbose' not in model_params:
        model_params['verbose'] = 200

    return CatBoostClassifier(**model_params)


def log_training_datasets(X_train, X_valid, y_train, y_valid, TARGET_COL, data_source, data_uri):
    """Log all training datasets to MLflow."""
    # Log original dataset (reconstructed)
    if data_source and data_uri:
        # This is a simplified reconstruction - in practice you'd want the original df
        print("⚠️  Original dataset logging skipped (would need full original dataframe)")
    
    # Log train/validation splits
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    log_dataset_to_mlflow(train_df, data_source or "processed", data_uri or "train_split", context="train")
    
    valid_df = X_valid.copy() 
    valid_df[TARGET_COL] = y_valid
    log_dataset_to_mlflow(valid_df, data_source or "processed", data_uri or "valid_split", context="validation")


def log_model_parameters(config, hyperparams, X_train, X_valid):
    """Log model parameters and training info to MLflow."""
    import mlflow
    params = {}
    
    if hyperparams:
        # For hyperparameter tuning
        params.update({
            'n_estimators': hyperparams['n_estimators'],
            'learning_rate': hyperparams['learning_rate'],
            'depth': hyperparams['depth'],
            'l2_leaf_reg': hyperparams['l2_leaf_reg']
        })
    else:
        # For standard training
        params.update({
            'n_estimators': config['model']['n_estimators'],
            'learning_rate': config['model']['learning_rate'],
            'depth': config['model']['depth'],
            'l2_leaf_reg': config['model']['l2_leaf_reg']
        })
    
    # Common parameters
    params.update({
        'random_seed': config['model']['random_seed'],
        'use_gpu': config['model']['use_gpu'],
        'test_size': config['training']['test_size'],
        'train_samples': len(X_train),
        'valid_samples': len(X_valid)
    })
    
    mlflow.log_params(params)


def train_and_evaluate_model(model, train_pool, valid_pool, config, X_train, X_valid, y_train, y_valid, mlflow_enabled=True):
    """Train model and return evaluation metrics."""
    if mlflow_enabled:
        import mlflow
    # Train model
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=config['model']['early_stopping_rounds']
    )

    # Predictions and metrics
    valid_proba = model.predict_proba(valid_pool)[:, 1]
    roc = roc_auc_score(y_valid, valid_proba)
    pr_auc = average_precision_score(y_valid, valid_proba)
    
    # Target recall threshold
    target_recall = config['training']['target_recall_threshold']
    thr, r_at_thr, p_at_thr = target_recall_threshold(y_valid, valid_proba, target_recall)
    y_pred = (valid_proba >= thr).astype(int)
    
    # Calculate F1 at target recall threshold
    f1_at_threshold = 2 * p_at_thr * r_at_thr / (p_at_thr + r_at_thr) if (p_at_thr + r_at_thr) > 0 else 0

    # Confusion matrix components (calculate early for use in metrics calculations)
    from sklearn.metrics import accuracy_score
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()
    
    # Additional derived metrics
    accuracy = accuracy_score(y_valid, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (r_at_thr + specificity) / 2.0
    
    # Churner-specific metrics
    churner_recall = r_at_thr  # Same as recall, but explicitly named for churners (class 1)
    non_churner_recall = specificity  # Recall for non-churners (class 0)
    churner_precision = p_at_thr  # Precision for churners (class 1)
    non_churner_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Precision for non-churners (class 0)
    
    # Business-relevant metrics
    total_churners = int(tp + fn)  # Actual churners in validation set
    total_non_churners = int(tn + fp)  # Actual non-churners in validation set
    predicted_churners = int(tp + fp)  # Predicted churners
    predicted_non_churners = int(tn + fn)  # Predicted non-churners
    
    # Calculate churn rates
    actual_churn_rate = total_churners / len(y_valid) if len(y_valid) > 0 else 0.0
    predicted_churn_rate = predicted_churners / len(y_valid) if len(y_valid) > 0 else 0.0

    # Log metrics to MLflow (only if enabled)
    if mlflow_enabled:
        mlflow.log_metrics({
            'roc_auc': roc,
            'pr_auc': pr_auc,
            'f1_at_target_recall': f1_at_threshold,
            'target_recall_threshold': thr,
            'precision_at_target_recall': p_at_thr,
            'actual_recall_at_threshold': r_at_thr
        })

    # Print validation metrics
    print("Validation metrics")
    print("------------------")
    print(f"ROC-AUC:                {roc:0.4f}")
    print(f"PR-AUC:                 {pr_auc:0.4f}")
    print(f"Target recall:          {target_recall:.1%}")
    print(f"Threshold:              {thr:0.4f}")
    print(f"F1 score:               {f1_at_threshold:0.4f}")
    print(f"\nChurner Metrics (Class 1):")
    print(f"  Churner recall:       {churner_recall:0.4f} ({churner_recall:.1%})")
    print(f"  Churner precision:    {churner_precision:0.4f} ({churner_precision:.1%})")
    print(f"  Total churners:       {total_churners:,} ({actual_churn_rate:.1%} of validation)")
    print(f"  Predicted churners:   {predicted_churners:,} ({predicted_churn_rate:.1%} of validation)")
    print(f"\nNon-Churner Metrics (Class 0):")
    print(f"  Non-churner recall:   {non_churner_recall:0.4f} ({non_churner_recall:.1%})")
    print(f"  Non-churner precision:{non_churner_precision:0.4f} ({non_churner_precision:.1%})")
    print(f"  Total non-churners:   {total_non_churners:,}")
    print(f"\nOverall Performance:")
    print(f"  Accuracy:             {accuracy:0.4f} ({accuracy:.1%})")
    print(f"  Balanced accuracy:    {balanced_accuracy:0.4f}")
    print(f"\nConfusion Matrix @ {target_recall:.1%} recall threshold")
    print(confusion_matrix(y_valid, y_pred))
    print(f"\nClassification Report @ {target_recall:.1%} recall threshold")
    print(classification_report(y_valid, y_pred, digits=4))

    # Feature importance
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.get_feature_importance(train_pool, type="FeatureImportance")
    }).sort_values("importance", ascending=False)
    print("\nTop 20 features:")
    print(fi.head(20).to_string(index=False))

    # All metrics calculated above, ready for return
    
    return {
        # Primary metrics
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "threshold": thr,
        "f1_score": f1_at_threshold,
        "precision": p_at_thr,
        "recall": r_at_thr,
        
        # Churner-specific metrics (class 1)
        "churner_recall": churner_recall,
        "churner_precision": churner_precision,
        "non_churner_recall": non_churner_recall,
        "non_churner_precision": non_churner_precision,
        
        # Business metrics
        "actual_churn_rate": actual_churn_rate,
        "predicted_churn_rate": predicted_churn_rate,
        "total_churners": total_churners,
        "total_non_churners": total_non_churners,
        "predicted_churners": predicted_churners,
        "predicted_non_churners": predicted_non_churners,
        
        # Additional performance metrics
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        
        # Confusion matrix components
        "true_positives": int(tp),
        "false_positives": int(fp), 
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        
        # Sample counts
        "train_samples": len(X_train),
        "valid_samples": len(X_valid),
        "feature_count": len(X_train.columns)
    }, fi, y_pred


def log_model_artifacts(model, train_pool, X_train, y_pred, fi):
    """Log model and feature importance to MLflow."""
    import mlflow
    import mlflow.catboost
    from mlflow.models.signature import infer_signature
    
    # Create input signature for better model tracking
    signature = infer_signature(X_train, y_pred)
    
    # Log model to MLflow with comprehensive metadata
    model_info = mlflow.catboost.log_model(
        model, 
        "model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name="churn-catboost-model"  # Auto-register in MLflow Model Registry
    )
    
    # Log feature importance as artifact
    fi_path = "feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)
    
    # Get model information
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model_uri = f"models:/churn-catboost-model/latest"
    
    print(f"\n✅ Model logged to MLflow Model Registry")
    print(f"📍 Model URI: {model_uri}")
    print(f"🏷️  Registered Model: {registered_model_uri}")
    print(f"🆔 Run ID: {run_id}")
    
    # Clean up temporary files
    if os.path.exists(fi_path):
        os.remove(fi_path)
    
    return model_uri, registered_model_uri


def run_catboost_training(df, config, hyperparams=None, data_source=None, data_uri=None, mlflow_enabled=True):
    """
    Complete CatBoost training pipeline with optional MLflow logging.
    
    Args:
        df: Training dataframe
        config: Configuration dictionary
        hyperparams: Optional hyperparameters dict for tuning (overrides config)
        data_source: Data source type for logging
        data_uri: Data URI for logging
        mlflow_enabled: Whether to log to MLflow (default True)
        
    Returns:
        dict: Training metrics and model URIs
    """
    TARGET_COL = config['model']['target_col']
    
    # Prepare data
    X_train, X_valid, y_train, y_valid, cat_idx = prepare_data_for_training(df, config)
    
    # Create Pools
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx)
    
    # Create model
    model = create_catboost_model(config, hyperparams)
    
    # Log datasets (only if MLflow enabled)
    if mlflow_enabled:
        log_training_datasets(X_train, X_valid, y_train, y_valid, TARGET_COL, data_source, data_uri)
    
    # Log parameters (only if MLflow enabled)
    if mlflow_enabled:
        log_model_parameters(config, hyperparams, X_train, X_valid)
    
    # Train and evaluate
    metrics, fi, y_pred = train_and_evaluate_model(model, train_pool, valid_pool, config, X_train, X_valid, y_train, y_valid, mlflow_enabled)
    
    # Log model artifacts (only if MLflow enabled)
    if mlflow_enabled:
        model_uri, registered_model_uri = log_model_artifacts(model, train_pool, X_train, y_pred, fi)
    else:
        model_uri, registered_model_uri = None, None
    
    # Add URIs to metrics
    metrics.update({
        "model_uri": model_uri,
        "registered_model_uri": registered_model_uri
    })
    
    if hyperparams:
        # Return metrics only for SageMaker (doesn't need model object)
        return metrics
    else:
        # Return full objects for standard training
        return model, fi, metrics


def setup_mlflow_tracking(config):
    """Configure MLflow tracking to use the SageMaker MLflow tracking server."""
    import mlflow
    try:
        tracking_uri = config['mlflow']['tracking_uri']
        print(f"Using SageMaker MLflow tracking server: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = config['mlflow']['experiment_name']
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: {experiment_name}")
        
    except KeyError as e:
        raise ValueError(f"Missing required MLflow configuration key: {e}")


def preprocess_data(df, config):
    """Apply data preprocessing steps from config."""
    # Convert timedelta columns to days
    timedelta_cols = ['account_age_d_cliente', 'days_since_last_update_cliente']
    for col in timedelta_cols:
        if col in df.columns and col in config['features']['datetime_features']:
            # Check if column is already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column {col} is already numeric, skipping conversion")
                continue
            # Only apply dt accessor if it's actually a timedelta
            try:
                df[col] = df[col].dt.total_seconds() / 86400
            except AttributeError:
                print(f"Column {col} is not timedelta format, skipping dt conversion")

    # Convert datetime columns
    datetime_cols = [col for col in config['features']['datetime_features'] if 'date' in col]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Convert to integer (handle NaN values)
    for c in config['features']['int_features']:
        if c in df.columns:
            # Fill NaN values with -1 before converting to int
            df[c] = df[c].fillna(-1).astype(int)

    # Convert to float (handle NaN values)
    if 'float_features' in config['features']:
        for c in config['features']['float_features']:
            if c in df.columns:
                # Fill NaN values with 0.0 for float features
                df[c] = df[c].fillna(0.0).astype(float)

    if 'tenure_bucket' in df.columns:
        df['tenure_bucket'] = df['tenure_bucket'].astype(str)

    df = df.replace({None: np.nan})
    
    return df


def write_sagemaker_metrics(metrics):
    """Write comprehensive metrics for SageMaker hyperparameter tuning."""
    try:
        os.makedirs("/opt/ml/output/metrics", exist_ok=True)
        with open("/opt/ml/output/metrics/metrics.json", "w") as f:
            json.dump({
                # Primary metrics
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
                "specificity": metrics["specificity"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                
                # Churner-specific metrics (class 1)
                "churner_recall": metrics["churner_recall"],
                "churner_precision": metrics["churner_precision"],
                "non_churner_recall": metrics["non_churner_recall"],
                "non_churner_precision": metrics["non_churner_precision"],
                
                # Business metrics
                "actual_churn_rate": metrics["actual_churn_rate"],
                "predicted_churn_rate": metrics["predicted_churn_rate"],
                "total_churners": metrics["total_churners"],
                "total_non_churners": metrics["total_non_churners"],
                "predicted_churners": metrics["predicted_churners"],
                "predicted_non_churners": metrics["predicted_non_churners"],
                
                # Threshold
                "threshold": metrics["threshold"],
                
                # Confusion matrix
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "true_negatives": metrics["true_negatives"],
                "false_negatives": metrics["false_negatives"],
                
                # Sample info
                "train_samples": metrics["train_samples"],
                "valid_samples": metrics["valid_samples"],
                "feature_count": metrics["feature_count"]
            }, f)
        print("✅ SageMaker comprehensive metrics written to /opt/ml/output/metrics/metrics.json")
    except PermissionError:
        print("⚠️  Could not write SageMaker metrics (not running in SageMaker environment)")
    except Exception as e:
        print(f"⚠️  Could not write SageMaker metrics: {e}")