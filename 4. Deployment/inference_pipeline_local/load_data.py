"""
Load and prepare data for inference.
"""

import os
import sys

import numpy as np
import pandas as pd

# Reuse the data_loader from the training pipeline
TRAINING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "training_pipeline_simplified")
)
sys.path.insert(0, TRAINING_DIR)
from data_loader import load_data as _load_from_config

INFERENCE_TABLE = "dth_churn_ml_inference.inference_features"


def load_inference_data(input_file=None, config=None):
    """Load data from a file (CSV/Parquet) or from Redshift."""

    # Option 1: Load from file
    if input_file:
        print(f"Loading data from: {input_file}")

        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        elif input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            raise ValueError("Unsupported format. Use .csv or .parquet")

        print(f"Loaded {len(df):,} rows")
        return df

    # Option 2: Load from Redshift (requires VPN)
    if config is None:
        raise ValueError("config is required when loading from Redshift")

    print(f"Loading data from Redshift: {INFERENCE_TABLE}")

    config["data"]["source"] = "redshift"
    config["data"]["redshift_sql"] = (
        f"SELECT * FROM {INFERENCE_TABLE} "
        "WHERE iddim_date_fim >= CURRENT_DATE "
        "LIMIT 100000;"
    )

    df = _load_from_config(config)
    print(f"Loaded {len(df):,} rows")
    return df


def preprocess(df, config):
    """Apply same preprocessing as training."""
    print("Preprocessing data...")

    # Convert timedelta columns to days
    for col in ["account_age_d_cliente", "days_since_last_update_cliente"]:
        if col in df.columns:
            try:
                df[col] = df[col].dt.total_seconds() / 86400
            except AttributeError:
                pass  # Already numeric

    # Fill missing values for int features
    for col in config.get("features", {}).get("int_features", []):
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)

    # Fill missing values for float features
    for col in config.get("features", {}).get("float_features", []):
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)

    return df


def prepare_features(df, config, model=None):
    """Prepare features for CatBoost prediction."""
    print("Preparing features...")

    target_col = config.get("model", {}).get("target_col", "churn")

    # Drop target if present
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
        print(f"  Dropped target column: {target_col}")

    # Get expected features and categorical indices from model
    if model is not None and hasattr(model, "feature_names_"):
        expected_features = list(model.feature_names_)
        cat_indices = list(model.get_cat_feature_indices())
        print(f"  Model expects {len(expected_features)} features")
        print(f"  Model has {len(cat_indices)} categorical features")

        # Check for missing features
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            print(f"  WARNING: Missing features: {missing}")

        # Create output DataFrame with correct columns and types
        X = pd.DataFrame()
        for i, col in enumerate(expected_features):
            if col in df.columns:
                X[col] = df[col].values
            else:
                # Missing feature - use default value
                X[col] = 0

            # Convert to correct type based on model expectations
            if i in cat_indices:
                # Categorical: convert to string
                X[col] = X[col].fillna("<MISSING>").astype(str)
            else:
                # Numeric: convert to float
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)

        print(f"  Features: {X.shape[1]}")
        print(f"  Categorical: {len(cat_indices)}")
        return X, cat_indices

    # Fallback: no model available, detect features from config
    # Drop ID columns
    id_cols = config.get("features", {}).get("id_features", [])
    cols_to_drop = [c for c in id_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} ID columns")

    # Drop datetime columns
    date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if date_cols:
        df = df.drop(columns=date_cols)
        print(f"  Dropped {len(date_cols)} datetime columns")

    # Find categorical columns (object or category dtype)
    numerical = set(
        config.get("features", {}).get("int_features", []) +
        config.get("features", {}).get("float_features", [])
    )
    cat_cols = [
        c for c in df.columns
        if (df[c].dtype == "object" or str(df[c].dtype) == "category")
        and c not in numerical
    ]

    # Fill missing values in categorical columns
    for col in cat_cols:
        df[col] = df[col].fillna("<MISSING>").astype(str)

    # Get column indices for CatBoost
    cat_indices = [df.columns.get_loc(col) for col in cat_cols]

    print(f"  Features: {df.shape[1]}")
    print(f"  Categorical: {len(cat_indices)}")

    return df, cat_indices


# Quick test when run directly
if __name__ == "__main__":
    from load_model import load_config

    print("=" * 60)
    print("LOAD DATA TEST")
    print("=" * 60)

    config = load_config()

    # This will load from Redshift (requires VPN)
    # df = load_inference_data(config=config)

    # For testing without Redshift, create dummy data
    print("\nCreating dummy test data...")
    df = pd.DataFrame({
        "idconsumo": [1, 2, 3],
        "tipo_produto_actual": ["normal", "premium", "normal"],
        "tenure_days": [100, 200, 300],
        "age": [25.0, 35.0, 45.0],
    })

    print(f"Test data shape: {df.shape}")
    print(df.head())
