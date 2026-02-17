"""
Load a trained churn prediction model.

Usage:
    from load_model import load_model, load_config, load_threshold

    model = load_model("../training_pipeline_simplified/models")
    config = load_config("../training_pipeline_simplified/models")
    threshold = load_threshold("../training_pipeline_simplified/models")
"""

import json
import os
import pickle
import yaml


# Default paths
DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "training_pipeline_simplified", "models")
)
TRAINING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "training_pipeline_simplified")
)


def load_model(model_dir=DEFAULT_MODEL_DIR):
    """
    Load the trained CatBoost model from a directory.

    Args:
        model_dir: Directory containing model.pkl

    Returns:
        Trained CatBoost model
    """
    model_path = os.path.join(model_dir, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully")
    return model


def load_config(model_dir=DEFAULT_MODEL_DIR):
    """
    Load model config. Tries model directory first, falls back to training config.

    Args:
        model_dir: Directory containing config.json

    Returns:
        Configuration dictionary
    """
    # Try model directory first
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            return json.load(f)

    # Fall back to training config
    fallback = os.path.join(TRAINING_DIR, "config.yaml")
    print(f"Loading config from: {fallback}")
    with open(fallback, "r") as f:
        return yaml.safe_load(f)


def load_threshold(model_dir=DEFAULT_MODEL_DIR):
    """
    Load the optimal prediction threshold from training metrics.

    Args:
        model_dir: Directory containing metrics.json

    Returns:
        Threshold value (default: 0.5)
    """
    metrics_path = os.path.join(model_dir, "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        threshold = float(metrics.get("threshold", 0.5))
        print(f"Loaded threshold: {threshold:.4f}")
        return threshold

    print("No metrics.json found, using default threshold: 0.5")
    return 0.5


# Quick test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("LOAD MODEL TEST")
    print("=" * 60)

    model = load_model()
    config = load_config()
    threshold = load_threshold()

    print(f"\nModel type: {type(model).__name__}")
    print(f"Config keys: {list(config.keys())}")
    print(f"Threshold: {threshold}")
