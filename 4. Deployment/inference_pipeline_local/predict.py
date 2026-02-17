"""
Run churn predictions using a trained model.

Usage:
    from predict import predict, save_predictions

    # Run predictions
    scores = predict(model, X, cat_indices)

    # Apply threshold and save
    save_predictions(scores, threshold, output_file="predictions.csv")
"""

import numpy as np
import pandas as pd
from catboost import Pool


def predict(model, X, cat_indices):
    """
    Run model prediction.

    Args:
        model: Trained CatBoost model
        X: Feature DataFrame
        cat_indices: List of categorical column indices

    Returns:
        Array of churn probability scores (0-1)
    """
    print(f"Running predictions on {len(X):,} rows...")

    # Create CatBoost Pool with categorical features
    pool = Pool(X, cat_features=cat_indices)

    # Get probability of churn (class 1)
    scores = model.predict_proba(pool)[:, 1]

    print(f"Predictions complete")
    return scores


def save_predictions(scores, threshold=0.5, output_file="predictions.csv"):
    """
    Save predictions to CSV file.

    Args:
        scores: Array of probability scores
        threshold: Threshold for binary prediction
        output_file: Output CSV path

    Returns:
        DataFrame with predictions
    """
    # Apply threshold
    predictions = (scores >= threshold).astype(int)

    # Create output DataFrame
    output = pd.DataFrame({
        "score": scores,
        "prediction": predictions
    })

    # Save to CSV
    output.to_csv(output_file, index=False)

    # Print summary
    print(f"\nResults saved to: {output_file}")
    print(f"  Total: {len(output):,}")
    print(f"  Predicted churners: {predictions.sum():,} ({predictions.mean():.1%})")
    print(f"  Threshold used: {threshold:.4f}")

    return output


# Quick test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("PREDICT TEST")
    print("=" * 60)

    # Create dummy scores for testing
    print("\nCreating dummy test scores...")
    scores = np.array([0.1, 0.4, 0.6, 0.8, 0.9])

    print(f"Test scores: {scores}")

    # Save with default threshold
    output = save_predictions(scores, threshold=0.5, output_file="test_predictions.csv")

    print("\nOutput DataFrame:")
    print(output)
