
# 📘 Quick Guide — `monitoring_exercise.ipynb`

This guide helps you configure **data capture**, **data quality**, **bias**, **explainability (SHAP)**, and **temporal drift** monitoring for a deployed SageMaker endpoint.

Use it alongside the user notebook. Anywhere you see **`# <- TODO ✏️`** in the notebook, fill in the value according to the instructions below.

---

## 1) Prerequisites

- Run inside **SageMaker Studio (Code Editor)** with a Python 3.x kernel.  
- Your model must already be **deployed** as an **Endpoint** (from your deployment notebooks).  
- IAM role attached to Studio / Execution Role with:
  - `sagemaker:*`, `s3:*`, `logs:*`, `events:*` (scoped to your buckets and prefixes).
- S3 buckets:
  - Data capture prefix (input + output logs)
  - Baseline dataset (CSV header) and **optional** Parquet fallback
  - Ground truth location (for bias monitor)

> **Tip:** Prefer a dedicated S3 prefix per endpoint and environment (e.g., `s3://<bucket>/capture/<env>/<endpoint>/`).

---

## 2) Configure the Notebook (one cell)

In the **Configuration cell**, update the following:

| Section | Key | Description | Example |
|---|---|---|---|
| `deployment` | `endpoint_name` | Existing endpoint to monitor | `my-prod-endpoint` |
| `deployment.data_capture` | `enable` | Toggle capture | `True` |
|  | `sampling_percentage` | % of requests to capture (1–100) | `50` |
|  | `s3_prefix` | S3 path where capture files land | `s3://my-bucket/data-capture/my-prod-endpoint/` |
|  | `content_type` | `text/csv` or `application/json` (must match inference contract) | `application/json` |
| `monitoring` | `enable` | Toggle monitors | `True` |
|  | `instance_type` | Instance for monitoring jobs | `ml.m5.large` |
|  | `schedule_cron` | When to run (CloudWatch cron) | `cron(0 2 * * ? *)` |
|  | `baseline_dataset_uri` | CSV with header for **baseline** | `s3://bucket/baseline/baseline.csv` |
|  | `ground_truth_s3_uri` | Ground-truth CSV/JSON location (for Bias) | `s3://bucket/labels/` |
| `clarify` | `enable_bias` | Enable Clarify **Bias** monitor | `True` |
|  | `enable_explainability` | Enable Clarify **SHAP** monitor | `True` |
|  | `label` | Target/label column name | `churn` |
|  | `facet_cols` | Protected attrs list | `["sex","age_bucket"]` |
|  | `headers` | Inference request headers (CSV capture) | `["f1","f2","f3","ts"]` |
|  | `predictor_config.content_type` | Request content type | `application/json` |
|  | `predictor_config.accept_type` | Response accept type | `application/json` |
|  | `predictor_config.probability_attribute` | JSON key for probability (if applicable) | `probabilities` |
|  | `explainability.shap_baseline_rows` | Sample size for SHAP | `200` |
|  | `positive_class` | Positive class value for bias metrics | `1` |

> **CSV vs JSON:**  
> - **CSV**: Fill `clarify.headers` to map captured rows back to feature names.  
> - **JSON**: Set `probability_attribute` if your output is a dict with probabilities.

---

## 3) Enable / Update Data Capture

Run the **Data Capture** cell to attach (or update) capture configuration on your endpoint:
- Verifies destination **S3 prefix**
- Enables request/response capture
- Sampling percentage as configured

**Common pitfalls:**
- Wrong `content_type` (must match your endpoint).  
- Missing S3 permissions (role/bucket policy).

---

## 4) Data Quality Monitoring

The **DefaultModelMonitor** will:
- (Optional) Generate **baseline** stats/constraints from your CSV baseline.
- Create a **schedule** to compare captured data against baseline.
- Write reports to `s3://<default-bucket>/sagemaker/.../monitoring/`

**Recommendations:**
- Use a representative **baseline** from recent production-like data.  
- Keep **feature order** and **types** consistent with inference.

---

## 5) Bias Monitoring (Clarify)

Requires **ground-truth** and `clarify.label`. You must provide:
- Ground-truth S3 (CSV/JSON) aligned by ID/timestamp (your data-mart).
- **Facet columns** (e.g., `sex`, `age_bucket`) for fairness slices.
- **Positive class** value (e.g., `1`).

Outputs fairness metrics (e.g., demographic parity, equal opportunity) for the configured facet(s).

**Tips:**
- Start with 1–2 facets, then expand.  
- Ensure your ground-truth refresh cadence matches the monitoring schedule.

---

## 6) Explainability Monitoring (Clarify SHAP)

- Creates SHAP-based explainability reports on a schedule.  
- Use it to monitor **feature attribution drift** (are top features changing?).  
- Control cost via `shap_baseline_rows` and schedule frequency.

**Note:** For tree models (XGBoost/CatBoost), SHAP is efficient. For deep models, expect higher cost.

---

## 7) Temporal Drift Checks (in-notebook PSI & KS)

The notebook includes a lightweight comparison between the **baseline** and a recent **captured** sample:
- Detects a **timestamp** column in baseline.  
- Compares up to 10 overlapping numeric features.  
- Computes: **PSI** (Population Stability Index) and **KS** (Kolmogorov–Smirnov).  
- Outputs `temporal_drift_summary.csv` in your run `ARTIFACT_DIR`.

**Interpretation (rules of thumb):**
- **PSI**: `< 0.1` stable, `0.1–0.25` moderate shift, `> 0.25` significant drift.  
- **KS**: `> 0.1` may indicate drift (depends on sample size).

**Make it work:**
- Ensure the baseline has a **timestamp** column.  
- For **CSV capture**, provide `clarify.headers` to parse request rows.  
- For **JSON capture**, the notebook will attempt to flatten payloads.

---

## 8) Inspect Schedules & Executions

The final section lists **schedules** and their **latest execution** status (Completed/Failed).  
Use SageMaker Studio **Experiments & Monitoring** panel or CloudWatch Logs for detailed traces.

---

## 9) Clean Up (optional)

In the **Cleanup** cell (commented):
- Delete monitoring schedules if needed.  
- Stop data capture to reduce storage.  
- Remove S3 artifacts when you’re done.

> **Cost control:** Lower sampling %, extend cron frequency, and downsize monitoring instance types.

---

## 10) Troubleshooting

- **AccessDenied**: Check role permissions and S3 bucket ACLs.  
- **Baseline errors**: Ensure CSV has header row; types align with inference.  
- **Bias monitor “missing label”**: Set `clarify.label` and provide ground truth.  
- **SHAP failed**: Reduce `shap_baseline_rows` or choose a larger instance type.  
- **No capture files**: Verify endpoint is receiving traffic and data capture is enabled.

---

## ✅ User Checklist

- [ ] Set **endpoint name**  
- [ ] Choose **capture prefix** and **content type**  
- [ ] Provide **baseline CSV** (with header)  
- [ ] (Bias) Set **label**, **positive class**, and **ground-truth URI**  
- [ ] (Bias) Choose **facet columns**  
- [ ] (Explainability) Set **SHAP sample size**  
- [ ] (Temporal drift) Ensure **timestamp** and **headers** mapping  
- [ ] Tune **cron** & **instance type**  
- [ ] Run all sections and verify **schedules** are active

Happy monitoring! 📈
