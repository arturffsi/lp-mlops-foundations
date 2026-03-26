"""
Simple Local Inference Script for Churn Prediction

This script combines all 3 steps:
  1. load_model.py - Load the trained model
  2. load_data.py  - Load and prepare data
  3. predict.py    - Run predictions

Usage:
  python infer.py                           # Load from Redshift, local model
  python infer.py --input data.parquet      # Load from file
  python infer.py --from-registry           # Download model from SageMaker
"""

import argparse
import os
from datetime import date

import boto3

# Import from our modules
from load_model import load_model, load_config, load_threshold, download_from_registry, AWS_PROFILE, AWS_REGION
from load_data import load_inference_data, preprocess, prepare_features
from predict import predict, save_predictions

S3_OUTPUT_BUCKET = "sagemaker-af-south-1-733246370304"
S3_OUTPUT_PREFIX = "learningpods/inference_output"


def main():
    parser = argparse.ArgumentParser(description="Churn Prediction Inference")
    parser.add_argument("--input", default="", help="Input file (.csv or .parquet)")
    parser.add_argument("--model-dir", default="", help="Model directory")
    parser.add_argument("--from-registry", action="store_true",
                        help="Download latest model from SageMaker Model Registry")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV")
    parser.add_argument("--upload-s3", action="store_true",
                        help="Upload predictions to S3 after saving locally")
    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("CHURN PREDICTION INFERENCE")
    print("=" * 60)

    # Step 1: Load model
    print("\n[Step 1] Loading model...")

    if args.from_registry:
        # Download from SageMaker Model Registry
        model_dir = download_from_registry("./models")
        model = load_model(model_dir)
        config = load_config(model_dir)
        threshold = load_threshold(model_dir)
    elif args.model_dir:
        model = load_model(args.model_dir)
        config = load_config(args.model_dir)
        threshold = load_threshold(args.model_dir)
    else:
        model = load_model()
        config = load_config()
        threshold = load_threshold()

    # Step 2: Load data
    print("\n[Step 2] Loading data...")
    if args.input:
        df = load_inference_data(input_file=args.input)
    else:
        df = load_inference_data(config=config)

    # Step 3: Preprocess and prepare features
    print("\n[Step 3] Preparing features...")
    df = preprocess(df, config)
    X, cat_indices = prepare_features(df, config, model=model)

    # Step 4: Predict
    print("\n[Step 4] Running predictions...")
    scores = predict(model, X, cat_indices)

    # Step 5: Save results
    print("\n[Step 5] Saving results...")
    save_predictions(scores, threshold, args.output)

    # Step 6: Upload to S3 (optional)
    if args.upload_s3:
        print("\n[Step 6] Uploading to S3...")
        s3_key = f"{S3_OUTPUT_PREFIX}/{date.today()}/{args.output}"
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        session.client("s3").upload_file(args.output, S3_OUTPUT_BUCKET, s3_key)
        print(f"  Uploaded to: s3://{S3_OUTPUT_BUCKET}/{s3_key}")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
