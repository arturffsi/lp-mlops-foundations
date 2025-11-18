
#!/usr/bin/env python3
"""
Minimal SageMaker inference handler (user edition)
-----------------------------------------------------
Works with a simple **artifact flow** where your model is bundled inside `model.tar.gz` and
extracted to `/opt/ml/model` by SageMaker.

✅ What this supports out of the box
  - CatBoost (.cbm)
  - XGBoost Booster (.json/.bst)
  - scikit-learn / pipelines (.pkl/.joblib)

👀 Where to customize (search for: # <- TODO ✏️)
  - Input format (JSON/CSV schema)
  - Feature ordering / renaming
  - Post-processing & thresholding
  - Selecting which outputs you return

This file implements the standard SageMaker model server hooks:
  - model_fn(model_dir)
  - input_fn(request_body, content_type)
  - predict_fn(input_data, model)
  - output_fn(prediction, accept)
"""

import os
import io
import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------
# Utilities
# ------------------------------
def _find_first(model_dir: str, exts: List[str]) -> str:
    for root, _, files in os.walk(model_dir):
        for f in files:
            lf = f.lower()
            for e in exts:
                if lf.endswith(e):
                    return os.path.join(root, f)
    return ""

def _load_catboost(path: str):
    from catboost import CatBoostClassifier, CatBoost
    m = CatBoost()
    m.load_model(path)
    return m

def _load_xgboost(path: str):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(path)
    return booster

def _load_sklearn(path: str):
    import joblib
    return joblib.load(path)

# ------------------------------
# SM: model_fn
# ------------------------------
def model_fn(model_dir: str):
    """
    Load the trained model. SageMaker passes model_dir=/opt/ml/model.
    """
    logger.info(f"Loading model from: {model_dir}")
    # 1) Allow explicit filename via env var
    explicit = os.getenv("MODEL_FILENAME", "")
    if explicit:
        full = os.path.join(model_dir, explicit)
        if not os.path.exists(full):
            raise FileNotFoundError(f"MODEL_FILENAME not found: {full}")
        return _load_by_extension(full)

    # 2) Auto-discovery by common extensions
    # CatBoost
    cbm = _find_first(model_dir, [".cbm"])
    if cbm:
        logger.info(f"Detected CatBoost model: {cbm}")
        return _load_catboost(cbm)

    # XGBoost
    xgbp = _find_first(model_dir, [".json", ".bst"])
    if xgbp:
        logger.info(f"Detected XGBoost Booster: {xgbp}")
        return _load_xgboost(xgbp)

    # scikit-learn
    skl = _find_first(model_dir, [".pkl", ".joblib"])
    if skl:
        logger.info(f"Detected sklearn model: {skl}")
        return _load_sklearn(skl)

    raise ValueError("No supported model artifact found in model_dir. "
                     "Expected one of: .cbm, .json/.bst, .pkl/.joblib")

def _load_by_extension(path: str):
    lp = path.lower()
    if lp.endswith(".cbm"):
        return _load_catboost(path)
    if lp.endswith(".json") or lp.endswith(".bst"):
        return _load_xgboost(path)
    if lp.endswith(".pkl") or lp.endswith(".joblib"):
        return _load_sklearn(path)
    raise ValueError(f"Unsupported model file: {path}")

# ------------------------------
# SM: input_fn
# ------------------------------
def input_fn(request_body: bytes, content_type: str):
    """
    Parses the request into a pandas.DataFrame.
    Supports JSON (list[dict]) and CSV with header.
    """
    content_type = (content_type or "application/json").lower()
    if "json" in content_type:
        payload = json.loads(request_body.decode("utf-8"))
        # Expecting a list of records/dicts. Examples:
        # [{"feature1": 1, "feature2": "A"}, {"feature1": 2, "feature2": "B"}]
        # <- TODO ✏️ If you expect a single dict, normalize to list here.
        if isinstance(payload, dict):
            payload = [payload]   # <- TODO ✏️ customise if needed
        df = pd.DataFrame(payload)
    elif "csv" in content_type or "text/plain" in content_type:
        df = pd.read_csv(io.StringIO(request_body.decode("utf-8")))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    # <- TODO ✏️ If your model requires fixed feature order or renaming:
    # expected_cols = ["age", "tenure_months", "monthly_charges", "contract_type", "country"]
    # df = df[expected_cols]

    # <- TODO ✏️ Type coercions / categorical normalisation if needed:
    # for c in ["contract_type", "country"]:
    #     df[c] = df[c].astype(str).str.strip().str.lower()

    return df

# ------------------------------
# SM: predict_fn
# ------------------------------
def predict_fn(input_data: pd.DataFrame, model: Any):
    """
    Runs inference and returns a Python object (will be serialized by output_fn).
    Tries to return probabilities if available, else predictions.
    """
    # CatBoost
    try:
        from catboost.core import CatBoost
        if isinstance(model, CatBoost):
            # <- TODO ✏️ For classifiers you may prefer probabilities over labels:
            # If your CatBoost model is a classifier, predict_proba works:
            try:
                proba = model.predict_proba(input_data)
                # Return probability of positive class if binary
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    scores = proba[:, 1].tolist()
                    return {"probabilities": proba.tolist(), "scores": scores}
                return {"probabilities": proba.tolist()}
            except Exception:
                preds = model.predict(input_data)
                return {"predictions": preds.tolist()}
    except Exception:
        pass

    # XGBoost Booster
    try:
        import xgboost as xgb
        if hasattr(model, "predict") and hasattr(model, "save_config"):
            dmatrix = xgb.DMatrix(input_data)
            preds = model.predict(dmatrix)
            # <- TODO ✏️ If classification probabilities are needed, ensure your Booster objective outputs probs.
            return {"predictions": preds.tolist()}
    except Exception:
        pass

    # scikit-learn
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)
        out = {"probabilities": proba.tolist()}
        # Provide positive-class score if binary
        if proba.ndim == 2 and proba.shape[1] >= 2:
            out["scores"] = proba[:, 1].tolist()
        return out
    if hasattr(model, "decision_function"):
        scores = model.decision_function(input_data)
        return {"scores": np.asarray(scores).ravel().tolist()}
    if hasattr(model, "predict"):
        preds = model.predict(input_data)
        return {"predictions": np.asarray(preds).ravel().tolist()}

    raise ValueError("Model type unsupported by default predict path. "
                     "Add custom logic in predict_fn. # <- TODO ✏️")

# ------------------------------
# SM: output_fn
# ------------------------------
def output_fn(prediction: Dict[str, Any], accept: str):
    accept = (accept or "application/json").lower()
    # <- TODO ✏️ If you want to return only selected fields:
    # prediction = {"scores": prediction.get("scores", [])}
    body = json.dumps(prediction)
    if "json" in accept or "*/*" in accept:
        return body, "application/json"
    if "csv" in accept:
        # Simple CSV writer for flat outputs
        if "predictions" in prediction:
            csv = "\n".join(map(str, prediction["predictions"]))
        elif "scores" in prediction:
            csv = "\n".join(map(str, prediction["scores"]))
        else:
            csv = "\n".join(map(str, prediction.get("probabilities", [])))
        return csv, "text/csv"
    # Default to JSON
    return body, "application/json"
