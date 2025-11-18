"""
Simplified Training Utilities for SageMaker
Clean, beginner-friendly functions for training churn models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, accuracy_score
)
from catboost import CatBoostClassifier, Pool


def preprocess_data(df, config):
    """
    Clean and prepare data for training.
    Handles type conversions and missing values.

    Args:
        df: Raw dataframe
        config: Configuration dictionary

    Returns:
        Processed dataframe
    """
    print("🔧 Preprocessing data...")

    # Convert timedelta columns to numeric (days)
    timedelta_cols = ['account_age_d_cliente', 'days_since_last_update_cliente']
    for col in timedelta_cols:
        if col in df.columns:
            try:
                df[col] = df[col].dt.total_seconds() / 86400  # Convert to days
            except AttributeError:
                pass  # Already numeric

    # Convert datetime columns
    for col in config['features']['datetime_features']:
        if col in df.columns and 'date' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert to int (fill NaN with -1)
    for col in config['features']['int_features']:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)

    # Convert to float (fill NaN with 0.0)
    for col in config['features']['float_features']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)

    # Handle None values
    df = df.replace({None: np.nan})

    print(f"✅ Preprocessing complete")
    return df


def prepare_features(df, config):
    """
    Prepare features for training.
    Removes ID columns, handles dates, separates X and y.

    Args:
        df: Preprocessed dataframe
        config: Configuration dictionary

    Returns:
        X_train, X_valid, y_train, y_valid, categorical_indices
    """
    print("\n🎯 Preparing features...")

    target_col = config['model']['target_col']

    # Drop ID columns
    id_cols_to_drop = [col for col in config['features']['id_features'] if col in df.columns]
    df = df.drop(columns=id_cols_to_drop)
    print(f"   Dropped {len(id_cols_to_drop)} ID columns")

    # Separate target from features
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])

    # Drop datetime columns (CatBoost can't handle them directly)
    date_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    if date_cols:
        X = X.drop(columns=date_cols)
        print(f"   Dropped {len(date_cols)} datetime columns")

    # Identify categorical features (exclude numerical ones)
    numerical_features = set(
        config['features'].get('int_features', []) +
        config['features'].get('float_features', [])
    )

    cat_cols = [
        c for c in X.columns
        if (X[c].dtype == 'object' or str(X[c].dtype) == 'category')
        and c not in numerical_features
    ]

    # Handle missing values in categorical columns
    for col in cat_cols:
        X[col] = X[col].fillna('<MISSING>').astype(str)

    # Get categorical column indices (needed by CatBoost)
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]

    print(f"   Features: {X.shape[1]} total, {len(cat_cols)} categorical")
    print(f"   Target distribution: {np.bincount(y)}")

    # Split into train/validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['model']['random_seed'],
        stratify=y  # Keeps same churn rate in both sets
    )

    print(f"   Train: {len(X_train):,} rows (churn rate: {y_train.mean():.2%})")
    print(f"   Valid: {len(X_valid):,} rows (churn rate: {y_valid.mean():.2%})")

    return X_train, X_valid, y_train, y_valid, cat_indices


def train_model(X_train, X_valid, y_train, y_valid, cat_indices, config):
    """
    Train CatBoost model.

    Args:
        X_train, X_valid: Feature matrices
        y_train, y_valid: Target vectors
        cat_indices: Indices of categorical columns
        config: Configuration dictionary

    Returns:
        Trained model
    """
    print("\n🚀 Training model...")
    print(f"   Model: CatBoost Classifier")
    print(f"   Iterations: {config['model']['n_estimators']}")
    print(f"   Learning rate: {config['model']['learning_rate']}")
    print(f"   Depth: {config['model']['depth']}")

    # Create CatBoost data structures
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_indices)

    # Create and train model
    model = CatBoostClassifier(
        iterations=config['model']['n_estimators'],
        learning_rate=config['model']['learning_rate'],
        depth=config['model']['depth'],
        l2_leaf_reg=config['model']['l2_leaf_reg'],
        random_seed=config['model']['random_seed'],
        eval_metric='AUC',
        verbose=100  # Print every 100 iterations
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=config['model']['early_stopping_rounds']
    )

    print(f"✅ Training complete (best iteration: {model.best_iteration_})")

    return model


def find_optimal_threshold(y_true, y_scores, target_recall):
    """
    Find prediction threshold that achieves target recall.

    Args:
        y_true: True labels
        y_scores: Prediction scores (probabilities)
        target_recall: Minimum recall we want to achieve

    Returns:
        threshold, actual_recall, precision
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Find thresholds that meet target recall
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid_idx) > 0:
        # Use the highest threshold that still meets target
        idx = valid_idx[-1]
        return thresholds[idx], recalls[idx], precisions[idx]
    else:
        # Can't achieve target recall
        return 0.0, 0.0, 0.0


def evaluate_model(model, X_valid, y_valid, config, train_size=None):
    """
    Evaluate model performance and print metrics.

    Args:
        model: Trained model
        X_valid, y_valid: Validation data
        config: Configuration dictionary

    Returns:
        Dictionary with all metrics
    """
    print("\n📊 Evaluating model...")

    # Get predictions
    y_proba = model.predict_proba(X_valid)[:, 1]

    # Find optimal threshold
    target_recall = config['training']['target_recall_threshold']
    threshold, recall, precision = find_optimal_threshold(y_valid, y_proba, target_recall)
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    roc_auc = roc_auc_score(y_valid, y_proba)
    pr_auc = average_precision_score(y_valid, y_proba)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = accuracy_score(y_valid, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

    # Print results
    print(f"\n{'='*60}")
    print(f"📈 VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"PR-AUC:               {pr_auc:.4f}")
    print(f"")
    print(f"Target Recall:        {target_recall:.1%}")
    print(f"Actual Recall:        {recall:.1%}")
    print(f"Precision:            {precision:.1%}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"Accuracy:             {accuracy:.1%}")
    print(f"Threshold:            {threshold:.4f}")
    print(f"")
    print(f"Confusion Matrix:")
    print(f"  True Negatives:     {tn:,}")
    print(f"  False Positives:    {fp:,}")
    print(f"  False Negatives:    {fn:,}")
    print(f"  True Positives:     {tp:,}")
    print(f"{'='*60}\n")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_valid.columns,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    print("🏆 Top 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    print()

    # Return metrics dictionary
    if train_size is None:
        # Fallback: approximate train size from validation size and test_size
        try:
            test_size = float(config['training']['test_size'])
            approx_train_size = int(len(y_valid) * (1 - test_size) / test_size)
        except Exception:
            approx_train_size = 0
    else:
        approx_train_size = int(train_size)

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'f1_score': float(f1),
        'recall': float(recall),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'threshold': float(threshold),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'churner_recall': float(recall),
        'churner_precision': float(precision),
        'train_samples': approx_train_size,
        'valid_samples': len(y_valid),
        'feature_count': len(X_valid.columns),
        'actual_churn_rate': float(y_valid.mean()),
        'predicted_churn_rate': float(y_pred.mean())
    }


def save_model(model, config, model_dir="/opt/ml/model"):
    """
    Save model for SageMaker deployment.

    Args:
        model: Trained CatBoost model
        config: Configuration dictionary
        model_dir: Directory to save model
    """
    import os
    import pickle
    import json

    print(f"\n💾 Saving model to: {model_dir}")

    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✅ Model saved")

    # Save config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"   ✅ Config saved")

    print(f"✅ All artifacts saved successfully\n")


def write_sagemaker_metrics(metrics):
    """
    Write metrics in SageMaker format for hyperparameter tuning.

    Args:
        metrics: Dictionary of metrics
    """
    import os
    import json

    try:
        os.makedirs("/opt/ml/output/metrics", exist_ok=True)
        with open("/opt/ml/output/metrics/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("✅ SageMaker metrics written")
    except Exception as e:
        print(f"⚠️  Could not write SageMaker metrics: {e}")
