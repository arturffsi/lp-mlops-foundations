
# ✅ Evaluation Guide — `evaluation_example.ipynb` & `evaluation_exercise.ipynb`

**Purpose:** Assess generalization via cross‑validation, calibrate thresholds (e.g., @target recall), and produce traceable metrics artifacts for validation & deployment. Designed for **SageMaker Studio (Code Editor)**.

---

## 🎯 Goals
- Measure **generalization** (StratifiedKFold / TimeSeriesSplit).
- Calibrate **operational threshold** (e.g., reach 0.80 recall target).
- Log results to **MLflow** and persist a compact `artifacts/metrics.json` used by validation.
- Support **CatBoost / XGBoost / scikit‑learn** models via `training_utils`.

---

## 🧱 Prereqs
- `config.yaml` with:
  - `data.source` (`parquet` or `redshift`) & URIs/SQL
  - `model.target_col`, `random_seed`, model hyperparams
  - `features.id_features` to drop from X
- Code files in the repo: `data_io.py`, `training_utils.py` (model factory + train/eval).
- Optional: **MLflow** tracking configured (SageMaker or local).

---

## 🔁 What the Example Notebook Does
1. **Load & preprocess** using `data_io.load_data` + `training_utils.preprocess_data`.
2. **Prepare features** (drop ID columns, handle categoricals, drop datetimes).
3. **Choose CV strategy**: `stratified` (default) or `timeseries`.
4. **Train & evaluate per fold** with `create_catboost_model` (or alt model in utils):
   - Metrics: ROC‑AUC, PR‑AUC, F1, **Recall**, Precision, chosen **threshold**.
5. **Aggregate** mean ± std; compute CI and a stability label.
6. **Save artifacts**:
   - `artifacts/metrics.json` (summary incl. recall@target)
   - (optional) `mlflow` nested runs with per‑fold metrics.
7. **Print summary**: table of folds + overall means/stds.

**Outputs → consumed by validation:**
- `artifacts/metrics.json` (primary)
- (optional) `artifacts/eval_report.md` for human notes

---

## ✏️ What to Customize in `evaluation_exercise.ipynb`
Look for **`# ← TODO ✏️`** markers to set:
- **Dataset & target**: `config['data']`, `model.target_col`
- **Model family**: CatBoost / XGBoost / Sklearn (swap factory or params)
- **CV strategy & folds**: `cv_strategy`, `cv_folds`
- **Primary metric & threshold target**: e.g., `target_recall = 0.80`
- **Logged metrics**: add/remove to MLflow
- **Artifact shape**: ensure `metrics.json` includes fields your validation expects

---

## 🧪 Quick Checklist
- [ ] Data loads and preprocessing completes
- [ ] CV runs with stable seeds and chosen strategy
- [ ] Threshold meets target (e.g., recall ≥ 0.80) or you know the gap
- [ ] `artifacts/metrics.json` written and version‑controlled in S3/run folder
- [ ] MLflow has per‑fold + parent summary (optional)

---

## 🛠️ Troubleshooting
- **Categorical NaNs / types** → ensure `<MISSING>` fill and `object` dtype before CatBoost/XGBoost Pools.
- **Datetime columns** in X → drop them (CatBoost JSON/Sklearn often can’t ingest `datetime64`).
- **Imbalanced data** → prefer PR‑AUC and recall‑centric thresholds; use class weights where supported.
