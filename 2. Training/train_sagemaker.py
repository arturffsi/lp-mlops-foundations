#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_sagemaker.py
Small/medium dataset trainer (loads full dataset in memory).

✅ Matches the notebook regex for metrics:
  - "Final ROC-AUC: <float>"
  - "Final F1 @ target recall: <float>"
  - "Churner recall: <float>"
  - "Churner precision: <float>"

✅ Arg names match Manual Search + Tuner:
  --n-estimators, --learning-rate, --depth, --l2-leaf-reg

Students pick the model via --model (catboost|xgboost|lgbm|logreg) and can
edit the TODOs to adapt features & preprocessing.
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Optional dependencies; import lazily inside builders when needed
# from catboost import CatBoostClassifier, Pool
# import xgboost as xgb
# import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import yaml

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

def read_parquet_uri(uri: str) -> pd.DataFrame:
    # <- TODO ✏️ If you want to support Redshift or CSV, add variants here.
    return pd.read_parquet(uri)

# -----------------------
# Feature preparation
# -----------------------

def prepare_dataframe(df: pd.DataFrame, target: str, date_col_candidates=None) -> tuple[pd.DataFrame, pd.Series, str]:
    date_col_candidates = date_col_candidates or ["iddim_date_inicio", "iddim_date_fim"]
    # <- TODO ✏️ Identify + drop IDs/leakage columns for YOUR dataset
    drop_cols = [
        target,
        "id", "customer_id", "codigocliente", "codigocontaservico",
        "iddim_contaservico_dth", "contract_number"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = df[target].astype(int).values
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    # Cast/clean basic types
    for c in X.columns:
        if np.issubdtype(X[c].dtype, np.datetime64):
            continue
        if X[c].dtype == "object":
            X[c] = X[c].astype("string")
    # Pick date column if available (used only for split)
    date_col = next((c for c in date_col_candidates if c in df.columns), None)
    return X, y, date_col

def build_sklearn_pipeline(cat_cols, num_cols, model_name, args):
    """
    Simple sklearn pipeline for logreg / xgboost / lgbm via one-hot.
    Students can adapt to target encoding, hashing, etc.
    """
    # One-hot for categoricals; passthrough numerics
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            n_jobs=-1,
            C=1.0  # <- TODO ✏️ you can expose C to tuner
        )
    elif model_name == "xgboost":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.depth,
            reg_lambda=args.l2_leaf_reg,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="auc",
            n_jobs=-1
        )
    elif model_name == "lgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.depth,
            reg_lambda=args.l2_leaf_reg,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            n_jobs=-1
        )
    else:
        raise ValueError("Unsupported sklearn pipeline model. Use logreg|xgboost|lgbm")

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def build_catboost_model(cat_idx, args):
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
# Train/Eval
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    # Generic controls
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--mlflow-mode", default="disabled", choices=["disabled", "local", "sagemaker"])
    parser.add_argument("--recall-target", default=0.80, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use-gpu", action="store_true")

    # Data sampling (kept for parity with notebook; not used to chunk in this small trainer)
    parser.add_argument("--sample-ratio", default=1.0, type=float)
    parser.add_argument("--chunk-size", default=50000, type=int)

    # Model choice & HPO params (names match tuner/manual)
    parser.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "lgbm", "logreg"])
    parser.add_argument("--n-estimators", dest="n_estimators", default=2000, type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", default=0.08, type=float)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--l2-leaf-reg", dest="l2_leaf_reg", default=3.0, type=float)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Load config
    cfg = load_config(args.config)
    parquet_uri = cfg["data"]["parquet_uri"]  # <- required in config.yaml
    target_col = cfg.get("columns", {}).get("target", "churn")  # <- TODO ✏️ align with your data

    # Load all data in memory (small/medium)
    df = read_parquet_uri(parquet_uri)
    if 0 < args.sample_ratio < 1.0:
        df = df.sample(frac=args.sample_ratio, random_state=args.seed).reset_index(drop=True)

    # Prepare
    X, y, date_col = prepare_dataframe(df, target_col)
    # Split: prefer time cut if date present else stratified shuffle
    if date_col:
        dates = parse_maybe_yyyymmdd(df[date_col])
        if dates.notna().mean() > 0.8:
            cutoff = dates.quantile(0.8)  # 80/20 split
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

    Xtr, ytr = X.loc[train_mask].copy(), y[train_mask]
    Xva, yva = X.loc[valid_mask].copy(), y[valid_mask]

    # Train depending on model
    if args.model == "catboost":
        # Prepare CatBoost categorical handling
        cat_cols = [c for c in Xtr.columns if Xtr[c].dtype in ["object", "string", "category"]]
        cat_idx = Xtr.columns.get_indexer(cat_cols).tolist()
        # Replace missing categoricals with '<MISSING>'
        for c in cat_cols:
            Xtr[c] = Xtr[c].astype("string").fillna("<MISSING>")
            Xva[c] = Xva[c].astype("string").fillna("<MISSING>")
        # Drop datetime columns (CatBoost can't take NaT)
        dt_cols = [c for c in Xtr.columns if np.issubdtype(Xtr[c].dtype, np.datetime64)]
        if dt_cols:
            Xtr = Xtr.drop(columns=dt_cols); Xva = Xva.drop(columns=dt_cols)

        from catboost import Pool
        train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
        valid_pool = Pool(Xva, yva, cat_features=cat_idx)

        model = build_catboost_model(cat_idx, args)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=200)
        proba = model.predict_proba(valid_pool)[:, 1]
    else:
        # Sklearn-style path with one-hot
        cat_cols = [c for c in Xtr.columns if Xtr[c].dtype in ["object", "string", "category"]]
        num_cols = [c for c in Xtr.columns if c not in cat_cols and not np.issubdtype(Xtr[c].dtype, np.datetime64)]
        # Drop datetimes
        dt_cols = [c for c in Xtr.columns if np.issubdtype(Xtr[c].dtype, np.datetime64)]
        if dt_cols:
            Xtr = Xtr.drop(columns=dt_cols); Xva = Xva.drop(columns=dt_cols)

        pipe = build_sklearn_pipeline(cat_cols, num_cols, args.model, args)
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xva)[:, 1]
        model = pipe  # persist the pipeline

    # Metrics
    roc = roc_auc_score(yva, proba)
    pr_auc = average_precision_score(yva, proba)
    thr, best_f1, p_at_thr, r_at_thr = best_f1_threshold(yva, proba)

    # Threshold at target recall
    prec, rec, thresholds = precision_recall_curve(yva, proba)
    target = float(args.recall_target)
    idx = np.where(rec[:-1] >= target)[0]
    if len(idx):
        i = idx[-1]
        t = float(thresholds[i]); recall_80 = float(rec[i]); precision_80 = float(prec[i])
    else:
        t, recall_80, precision_80 = 0.0, 0.0, 0.0

    y_pred = (proba >= t).astype(int)

    # Regex-friendly prints (DON'T CHANGE LINE TEXTS unless you change notebook regex)
    print(f"Final ROC-AUC: {roc:.4f}")
    print(f"Final F1 @ target recall: {best_f1:.4f}")
    print(f"Churner recall: {recall_80:.4f}")
    print(f"Churner precision: {precision_80:.4f}")

    # Optional: extra diagnostics
    print("\nClassification Report @ target recall threshold")
    print(classification_report(yva, y_pred, digits=4))

    # Save model artifact for SageMaker
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, out_path)
    print(f"Model artifact saved to: {out_path}")

if __name__ == "__main__":
    main()
