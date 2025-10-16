
# 🧪 Validation Guide — `validation_example.ipynb` & `validation_exercise.ipynb`

**Purpose:** Enforce **quality gates**, compare the **Candidate** vs **Champion** in SageMaker Model Registry, and decide promotion. Produces lineage for deployment. Designed for **SageMaker Studio (Code Editor)**.

---

## 🎯 Goals
- Define **quality gates** (e.g., `mean recall@target ≥ 0.80`, `std ≤ 0.02`).
- Resolve **Candidate** (from evaluation) and **Champion** (Registry).
- **Block promotion** if gates fail or Candidate < Champion.
- Verify **artifact completeness** (schema, splits, metrics, model‑info).
- Register Candidate (if passing) with **tags** for lineage (`candidate_run_id`, `data_version`, `metrics_hash`).

---

## 🧱 Prereqs
- `artifacts/metrics.json` from **evaluation**.
- Access to **Model Package Group** in SageMaker Model Registry.
- Model artifact (`model.tar.gz`) in S3 with contract:
  - `inference.py`, `requirements.txt`
  - `model/{model.cbm|model.json|model.pkl}`
  - `artifacts/feature_schema.json`, `artifacts/splits.json`, `artifacts/metrics.json`
  - `model-info.json` (candidate_run_id, hashes, data_version)

---

## 🔎 What the Example Notebook Does
1. **Load candidate metrics** from `artifacts/metrics.json`.
2. **Define gates** (defaults focus on **recall@target** and stability).
3. **Fetch Champion** from **Model Registry** (latest Approved in group).
4. **Compare** Candidate vs Champion on primary metric (tie‑breakers allowed).
5. **Check artifacts** exist and are readable.
6. **Decision**:
   - If **pass** → Register/Update package, tag with lineage, set Approval (per policy).
   - If **fail** → No registration or mark as Rejected; emit reasons.
7. **Write result**:
   - `validation_results.json` (decision, reasons, candidate_run_id, package_arn if any)
   - Attach tags for **deployment** discovery.

---

## ✏️ What to Customize in `validation_exercise.ipynb`
Look for **`# ← TODO ✏️`** markers to set:
- **Quality gates**: primary metric (e.g., recall@target), thresholds, std caps
- **Champion comparison rule**: must improve vs. allow within tolerance
- **Artifact requirements**: which files are mandatory
- **Model Package Group**: name, Approval policy (auto/pend/manual)
- **Tag keys/values**: `candidate_run_id`, `data_version`, `business_unit`, etc.
- **Fail‑open vs fail‑closed** policies

---

## 📤 Outputs → consumed by deployment
- `validation_results.json` with `candidate_run_id`, decision, and (optional) `model_package_arn`
- SageMaker **Model Package** with tags (if registered)
- S3 path to validated `model.tar.gz`

---

## 🧪 Quick Checklist
- [ ] Gates computed on candidate and **passed**
- [ ] Champion metrics loaded and compared
- [ ] Artifact contract verified (schema/splits/metrics/model‑info)
- [ ] Package registered (or decision recorded with reasons)
- [ ] `candidate_run_id` emitted for deployment discovery

---

## 🛠️ Troubleshooting
- **No Champion** → fall back to Candidate if first model; document policy.
- **Registry permission errors** → confirm role has Model Registry access.
- **Inconsistent metrics schemas** → keep `metrics.json` stable; version if evolving.
- **Artifact missing** → **fail fast**; evaluation must re‑publish complete artifacts.
