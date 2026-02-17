"""
Simple Local Inference Script for Churn Prediction

This script combines all 3 steps:
  1. load_model.py - Load the trained model
  2. load_data.py  - Load and prepare data
  3. predict.py    - Run predictions

Usage:
  python infer.py                           # Load from Redshift
  python infer.py --input data.parquet      # Load from file
  python infer.py --model-dir /path/to/model
"""

import argparse
import os

# Import from our modules
from load_model import load_model, load_config, load_threshold
from load_data import load_inference_data, preprocess, prepare_features
from predict import predict, save_predictions


def main():
    parser = argparse.ArgumentParser(description="Churn Prediction Inference")
    parser.add_argument("--input", default="", help="Input file (.csv or .parquet)")
    parser.add_argument("--model-dir", default="", help="Model directory")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV")
    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("CHURN PREDICTION INFERENCE")
    print("=" * 60)

    # Step 1: Load model
    print("\n[Step 1] Loading model...")
    if args.model_dir:
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

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
