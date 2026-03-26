"""
Load a trained churn prediction model.

Supports two sources:
  1. Local directory (from training_pipeline_simplified/models/)
  2. SageMaker Model Registry (downloads the latest approved model)
"""

import json
import os
import pickle
import tarfile
import tempfile

import boto3
import yaml


# Default paths
DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "training_pipeline_simplified", "models")
)
TRAINING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "training_pipeline_simplified")
)

# SageMaker Model Registry settings
MODEL_PACKAGE_GROUP = "learningpods-model-group"
AWS_REGION = "af-south-1"
AWS_PROFILE = "SageMaker-Full-AI-Access-733246370304"


def download_from_registry(model_dir="./models"):
    """Download the latest model from SageMaker Model Registry.

    Queries the model registry for the latest version, downloads model.tar.gz
    from S3, and extracts it to model_dir.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Connect to SageMaker
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    sm_client = session.client("sagemaker")
    s3_client = session.client("s3")

    # Get the latest approved model from the registry
    print(f"Querying model registry: {MODEL_PACKAGE_GROUP}")
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )

    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(f"No approved models found in {MODEL_PACKAGE_GROUP}")

    model_package_arn = packages[0]["ModelPackageArn"]
    version = packages[0].get("ModelPackageVersion", "?")
    print(f"  Latest approved version: {version}")

    # Get the S3 path to model.tar.gz
    details = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    s3_uri = details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    print(f"  Model artifact: {s3_uri}")

    # Parse S3 URI (s3://bucket/key)
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    # Download and extract model.tar.gz
    tar_path = os.path.join(model_dir, "model.tar.gz")
    print(f"  Downloading to {model_dir}/...")
    s3_client.download_file(bucket, key, tar_path)

    print(f"  Extracting model.tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)

    # Clean up the tar file
    os.remove(tar_path)

    print(f"  Done! Model files extracted to {model_dir}/")
    return model_dir


def load_model(model_dir=DEFAULT_MODEL_DIR):
    """Load the trained CatBoost model from a directory."""
    model_path = os.path.join(model_dir, "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully")
    return model


def load_config(model_dir=DEFAULT_MODEL_DIR):
    """Load model config. Tries model dir first, falls back to training config.yaml."""
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
    """Load the optimal prediction threshold from metrics.json (default: 0.5)."""
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
