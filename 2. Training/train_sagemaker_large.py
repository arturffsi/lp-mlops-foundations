#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_sagemaker_large.py
Large dataset trainer with streaming + downsampling.

Strategy:
- Iterates over a list of Parquet files from the S3 URI pattern
- Loads file-by-file (or row-group by row-group) and appends a DOWN-SAMPLED slice
  controlled by --sample-ratio and --chunk-size
- Trains the same models supported in the small trainer

Users can:
- Adjust the sampling strategy
- Swap models / add encoders
- Add Redshift reader if needed

✅ Metrics lines match regex expected by notebook.
✅ Hyperparameter arg names match Manual Search + Tuner.
"""

import argparse
import os
from pathlib import Path
import glob
import re

import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------
# Utils
# -----------------------

def best_f1_threshold(y_true, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    i = np.argmax(f1)
    return float(thresholds[i]), float(f1[i]), float(precisions[i]), float(recalls[i])

def parse_maybe_yyyymmdd(s):
    if pd.api.types.is_integer_dtype(s) or (s.dtype == object and s.astype(str).str.fullmatch(r"\d{8}").all()):
        return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def expand_s3_parquet_uris(uri_pattern: str) -> list[str]:
    """
    For simplicity, we support:
      - local glob patterns
      - s3://bucket/prefix/*.parquet  (requires s3fs installed)
    """
    if uri_pattern.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=False)
        # crude globbing over s3 prefix
        # e.g., s3://bucket/path/*.parquet -> (bucket, key_prefix)
        m = re.match(r"s3://([^/]+)/(.+)", uri_pattern)
        if not m:
            raise ValueError(f"Invalid S3 URI: {uri_pattern}")
        bucket, keypat = m.groups()
        # list then filter client-side
        prefix = keypat.split("*", 1)[0]
        candidates = fs.glob(f"{bucket}/{prefix}*")
        files = [f"s3://{p}" for p in candidates if p.endswith(".parquet")]
        if not files:
            raise FileNotFoundError(f"No parquet files found at {uri_pattern}")
        return sorted(files)
    else:
        files = glob.glob(uri_pattern)
        if not files:
            raise FileNotFoundError(f"No parquet files found at {uri_pattern}")
        return sorted(files)

def stream_sample_parquet(files: list[str], sample_ratio: float, chunk_size: int, seed: int) -> pd.DataFrame:
    """
    Iterates file-by-file; from each file, reads into pandas and takes a sample
    of at most `chunk_size` rows * sample_ratio (ceil), then concatenates.
    This is a pragmatic starter template; users can switch to pyarrow.dataset for
    row-group streaming.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for fp in files:
        df = pd.read_parquet(fp)
        if sample_ratio < 1.0:
            n = int(np.ceil(len(df) * sample_ratio))
            if n > 0:
                # If chunk_size is used to cap each file contribution
                n = min(n, chunk_size)
                idx = rng.choice(len(df), size=n, replace=False)
                df = df.iloc[idx].copy()
            else:
                continue
        else:
            # full file but capped by chunk_size per file to avoid blowups
            if len(df) > chunk_size:
                df = df.sample(n=chunk_size, random_state=seed)
        chunks.append(df)
    if not chunks:
        raise RuntimeError("No rows collected. Increase sample-ratio or check input.")
    out = pd.concat(chunks, ignore_index=True)
    return out

def build_sklearn_pipeline(cat_cols, num_cols, model_name, args):
    from sklearn.linear_model import LogisticRegression
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1)
    elif model_name == "xgboost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.depth,
            reg_lambda=args.l2_leaf_reg,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="auc", n_jobs=-1
        )
    elif model_name == "lgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.depth,
            reg_lambda=args.l2_leaf_reg,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary", class_weight="balanced", n_jobs=-1
        )
    else:
        raise ValueError("Unsupported sklearn pipeline model. Use logreg|xgboost|lgbm")
    return Pipeline([("pre", pre), ("clf", clf)])

def build_catboost_model(args):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=args.seed,
        auto_class_weights="Balanced",
        task_type="CPU" if not args.use_gpu else "GPU",
        verbose=200
    )
    return model

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--mlflow-mode", default="disabled", choices=["disabled", "local", "sagemaker"])
    parser.add_argument("--recall-target", default=0.80, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use-gpu", action="store_true")

    # Streaming controls
    parser.add_argument("--sample-ratio", default=0.25, type=float)  # smaller by default for large data
    parser.add_argument("--chunk-size", default=50000, type=int)

    # Model + HPO params (names match notebook)
    parser.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "lgbm", "logreg"])
    parser.add_argument("--n-estimators", dest="n_estimators", default=2000, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", default=0.08, type=float)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--l2-leaf-reg", dest="l2_leaf_reg", default=3.0, type=float)

    args = parser.parse_args()
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    parquet_uri = cfg["data"]["parquet_uri"]  # required
    target_col = cfg.get("columns", {}).get("target", "churn")  # <- TODO ✏️ align with your data

    # Expand the parquet pattern to a list of files and stream-sample
    files = expand_s3_parquet_uris(parquet_uri)
    df = stream_sample_parquet(files, args.sample_ratio, args.chunk_size, args.seed)

    # <- TODO ✏️ drop IDs/leakage columns for YOUR dataset if needed
    drop_cols = [target_col, "customer_id", "codigocliente", "codigocontaservico"]
    y = df[target_col].astype(int).values
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    # Basic cleanup
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("string")

    # Time-aware split if possible
    date_col = next((c for c in ["iddim_date_inicio", "iddim_date_fim"] if c in df.columns), None)
    if date_col:
        dates = parse_maybe_yyyymmdd(df[date_col])
        if dates.notna().mean() > 0.8:
            cutoff = dates.quantile(0.8)
            train_mask = dates < cutoff
            valid_mask = ~train_mask
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
            tr, va = next(splitter.split(X, y))
            train_mask = pd.Series(False, index=X.index); train_mask.iloc[tr] = True
            valid_mask = ~train_mask
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        tr, va = next(splitter.split(X, y))
        train_mask = pd.Series(False, index=X.index); train_mask.iloc[tr] = True
        valid_mask = ~train_mask

    Xtr, ytr = X.loc[train_mask], y[train_mask]
    Xva, yva = X.loc[valid_mask], y[valid_mask]

    if args.model == "catboost":
        # CatBoost path with native categorical support
        cat_cols = [c for c in Xtr.columns if Xtr[c].dtype in ["object", "string", "category"]]
        for c in cat_cols:
            Xtr[c] = Xtr[c].astype("string").fillna("<MISSING>")
            Xva[c] = Xva[c].astype("string").fillna("<MISSING>")
        dt_cols = [c for c in Xtr.columns if np.issubdtype(Xtr[c].dtype, np.datetime64)]
        if dt_cols:
            Xtr = Xtr.drop(columns=dt_cols); Xva = Xva.drop(columns=dt_cols)
        from catboost import Pool
        cat_idx = Xtr.columns.get_indexer(cat_cols).tolist()
        train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
        valid_pool = Pool(Xva, yva, cat_features=cat_idx)

        model = build_catboost_model(args)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=200)
        proba = model.predict_proba(valid_pool)[:, 1]
    else:
        # OHE pipeline path for xgboost/lgbm/logreg
        cat_cols = [c for c in Xtr.columns if Xtr[c].dtype in ["object", "string", "category"]]
        num_cols = [c for c in Xtr.columns if c not in cat_cols and not np.issubdtype(Xtr[c].dtype, np.datetime64)]
        dt_cols = [c for c in Xtr.columns if np.issubdtype(Xtr[c].dtype, np.datetime64)]
        if dt_cols:
            Xtr = Xtr.drop(columns=dt_cols); Xva = Xva.drop(columns=dt_cols)

        pipe = build_sklearn_pipeline(cat_cols, num_cols, args.model, args)
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xva)[:, 1]
        model = pipe

    # Metrics + regex lines
    roc = roc_auc_score(yva, proba)
    pr_auc = average_precision_score(yva, proba)
    thr, best_f1, p_at_thr, r_at_thr = best_f1_threshold(yva, proba)

    prec, rec, thresholds = precision_recall_curve(yva, proba)
    target = float(args.recall_target)
    idx = np.where(rec[:-1] >= target)[0]
    if len(idx):
        i = idx[-1]
        t = float(thresholds[i]); recall_target = float(rec[i]); precision_target = float(prec[i])
    else:
        t, recall_target, precision_target = 0.0, 0.0, 0.0

    print(f"Final ROC-AUC: {roc:.4f}")
    print(f"Final F1 @ target recall: {best_f1:.4f}")
    print(f"Churner recall: {recall_target:.4f}")
    print(f"Churner precision: {precision_target:.4f}")

    y_pred = (proba >= t).astype(int)
    print("\nClassification Report @ target recall threshold")
    print(classification_report(yva, y_pred, digits=4))

    # Save model
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, out_path)
    print(f"Model artifact saved to: {out_path}")

if __name__ == "__main__":
    main()
