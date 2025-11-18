
# 🚀 Deployment Guide — `deployment_example.ipynb` & `deployment_exercise.ipynb`

These two notebooks implement the **deployment stage** of your MLOps pipeline inside **Amazon SageMaker Studio (Code Editor)**. They take the model that passed your validation gates, package and register it (if needed), deploy it to a managed endpoint with **lineage tags**, and enable safe rollout.

> **TL;DR**
> - **deployment_example.ipynb**: a fully wired reference flow that *auto-discovers the Candidate (Challenger) model* using tags from `validation_example.ipynb` (e.g., `candidate_run_id`) and deploys it with best practices.
> - **deployment_exercise.ipynb**: a user version with **# ← TODO ✏️** markers to customize model lookup, endpoint config, rollout, and autoscaling.

---

## 🎯 Goals of the Deployment Stage

1. **Promote the right model**: Fetch the **Candidate** that **passed validation** (quality gates) and compare/confirm against the **Champion** if needed.
2. **Package & register** (if not already): Create a deterministic `model.tar.gz` that includes `inference.py`, `requirements.txt`, `model/…`, and `artifacts/…`, then upload to S3 and (optionally) the **Model Registry**.
3. **Create the hosted endpoint** using SageMaker best practices:
   - Proper image + model artifact wiring
   - **Lineage tags** (data version, run ids, package ARN, git hash, metrics hash, etc.)
   - **Safe rollout** (shadow/canary/blue‑green)
   - **Autoscaling** and **CloudWatch alarms**
4. **Prove it works**: Run a smoke test invoke and record inputs/outputs for traceability.
5. **Operate safely**: Provide a clear **rollback** path and cleanup options.

---

## 🧱 Prerequisites & Inputs

- ✅ You’ve run **`evaluation_example.ipynb`** and **`validation_example.ipynb`**.  
- ✅ The Candidate that passed validation was tagged in the Model Registry or in your artifact store (e.g., tag: `candidate_run_id=<RUN_ID>`).  
- ✅ You have (or can build) a **correctly structured** `model.tar.gz`:
  ```
  model.tar.gz
  ├── inference.py
  ├── requirements.txt
  ├── model/                 # one of: model.cbm | model.json/.bst | model.pkl/.joblib
  ├── artifacts/             # lineage/evaluation context
  │   ├── feature_schema.json
  │   ├── splits.json
  │   ├── metrics.json
  │   └── eval_report.md     # optional
  └── model-info.json        # includes candidate_run_id, data_version, hashes, tags
  ```
- ✅ SageMaker execution role has S3 + Model Registry + CloudWatch + Application Auto Scaling permissions.
- ✅ You know your **inference image** (e.g., BYOS, SKLearn, PyTorch, or a custom ECR image).

---

## 🗺️ High-Level Flow (both notebooks)

1. **Configuration & session setup**: region, role, default bucket, package group, endpoint name/version.
2. **Resolve the Candidate**:
   - Read `candidate_run_id` from validation outputs or **query the Model Registry** for the package tagged with it.
   - Retrieve S3 model artifact URI (`model.tar.gz`) and meta (`model-info.json`).
3. **Validate artifacts** (fast checks): ensure `inference.py`, `requirements.txt`, one valid `model/*` file, and the key files in `artifacts/` are present.
4. **Create/Update the SageMaker Model**:
   - Set `model_data` to S3 tarball and `image_uri` to the serving container.
   - Attach **lineage tags** (candidate_run_id, data_version, metrics hash, code ref, package ARN).
5. **Rollout strategy**:
   - **Shadow**: mirror production traffic with zero weight to Candidate.
   - **Canary**: shift 5–10% traffic to Candidate and progressively increase.
   - **Blue/Green**: deploy a parallel stack and switch over.
6. **Autoscaling & Alarms**:
   - Target-based scaling on invocations/latency.
   - CloudWatch alarms for 5xx, high latency, CPU/memory (if supported), or cost proxies.
7. **Smoke test**: invoke the endpoint with a small payload to validate readiness.
8. **Document & tag**: store deployment metadata (endpoint tags, config snapshot) for audits & rollback.
9. **(Optional) Cleanup**: helper cells to delete endpoint/resources in sandbox contexts.

---

## 📓 `deployment_example.ipynb` — What It Does

### Step-by-step
1. **Init & Config**
   - Set `REGION`, `ROLE`, `BUCKET`, `MODEL_PACKAGE_GROUP`, and a deterministic `ENDPOINT_NAME` with timestamp/hash.
2. **Discover the Candidate**
   - Read `candidate_run_id` from:
     - The **validation output** (e.g., `validation_results.json`, registry tags), or
     - Direct **Model Registry** query: latest package in group with tag `candidate_run_id=<id>` and `ApprovalStatus="Approved"` (or your policy).
3. **Fetch Artifacts**
   - Pull S3 URI for `model.tar.gz`; download **`model-info.json`** to confirm lineage: data version, metrics hash, git commit, etc.
4. **Sanity Checks**
   - Confirm required files exist inside the tar. If not, **fail fast** (don’t deploy partial artifacts).
5. **Create the SageMaker Model**
   - Use `sagemaker.model.Model` (or framework-specific variant) with `image_uri` and `model_data` pointing to S3 tarball.  
   - **Attach tags**:
     - `candidate_run_id`, `data_version`
     - `model_package_arn` (if from Registry)
     - `metrics_hash`, `code_ref`, `build_hash`
6. **Endpoint Config & Deployment**
   - Choose strategy (default: direct / canary).  
   - For **canary**, create two variants (Champion/Candidate) and split traffic (e.g., 90%/10%).
   - Create endpoint or update existing endpoint. Wait for `InService`.
7. **Autoscaling**
   - Register scalable target (variant) and put scaling policy (e.g., target invocations-per-instance).
8. **Data Capture**
   - Enable capture to S3; configure sampling (e.g., 100% or 20%).  
9. **Smoke Test**
   - Call the endpoint using the notebook’s sample payload; verify 200 and capture prediction.
10. **Summary & Links**
   - Print endpoint info, tags, CloudWatch links, and S3 capture location.


### Best Practices Applied
- **Artifact contract**: inference handler + requirements + deterministic tar layout.
- **Lineage**: endpoint and model tags for auditability.
- **Progressive rollout** with optional **shadow** first.
- **Autoscaling** and **alarms** to keep SLOs and cost in check.
- **Data capture** for downstream drift/bias checks.


---

## ✍️ `deployment_exercise.ipynb` — What *You* Customize

This version contains **# ← TODO ✏️** markers. Users must configure the following:

1. **Model lookup**
   - `MODEL_PACKAGE_GROUP`  **# ← TODO ✏️**  
   - `CANDIDATE_TAG_KEY` / `CANDIDATE_TAG_VALUE` (e.g., `candidate_run_id`) **# ← TODO ✏️**  
   - Whether to require `ApprovalStatus="Approved"` or allow pending **# ← TODO ✏️**

2. **Endpoint configuration**
   - `ENDPOINT_NAME` naming convention **# ← TODO ✏️**  
   - `INSTANCE_TYPE`, `INITIAL_INSTANCE_COUNT` **# ← TODO ✏️**  
   - Rollout mode: `shadow` / `canary` / `blue_green` **# ← TODO ✏️**  
   - Traffic weights for variants **# ← TODO ✏️**

3. **Lineage / tags**
   - Which tags to apply: `data_version`, `metrics_hash`, `git_commit`, `business_unit`, etc. **# ← TODO ✏️**

4. **Autoscaling & Alarms**
   - Scaling target (InvocationsPerInstance / Latency) **# ← TODO ✏️**  
   - Min/max capacity **# ← TODO ✏️**  
   - CloudWatch alarms (p99 latency, 5xx rate) thresholds **# ← TODO ✏️**

5. **Smoke test payload**
   - Input schema & example rows **# ← TODO ✏️**  
   - Response contract (scores vs probabilities vs predictions) **# ← TODO ✏️**

6. **Rollback policy**
   - Criteria for pausing/canceling rollout **# ← TODO ✏️**  
   - Revert action (switch traffic, delete candidate variant, restore champion) **# ← TODO ✏️**


### Exercise Checklist
- [ ] Resolved Candidate via `candidate_run_id`  
- [ ] Verified artifact contract (`inference.py`, `requirements.txt`, `model/*`, `artifacts/*`)  
- [ ] Chosen rollout strategy & traffic weights  
- [ ] Added lineage tags to Model **and** Endpoint  
- [ ] Enabled autoscaling and alarms  
- [ ] Performed smoke test and recorded results  
- [ ] Documented rollback plan


---

## 🧯 Troubleshooting

- **Endpoint stuck in “Creating”** → Check IAM role, image URI, and S3 permissions for **model data** and **data capture** paths.
- **Container errors (ExitCode != 0)** → Inspect CloudWatch logs; ensure `inference.py` is at the **tar root**, `requirements.txt` imports succeed, and the model path exists.
- **Model not found** → Confirm `model-info.json` and that the file in `/model` matches extensions expected by your handler.
- **Throttling/scale** → Increase min capacity or lower target value; verify quotas for the instance type in region.
- **Rollout did not shift traffic** → Re-check variant weights and endpoint config revision; note that updates are asynchronous.


---

## 📌 Key Concepts (Glossary)

- **Champion**: The currently approved model serving production traffic.  
- **Challenger (Candidate)**: A validated model attempting promotion.  
- **Model Package Group**: Registry logical group used for versioning and approvals.  
- **Lineage Tags**: Endpoint/Model tags tying deployment to data, code, and runs (e.g., `candidate_run_id`).  
- **Shadow / Canary / Blue‑Green**: Progressive rollout strategies to reduce risk.  


---

## ✅ Outcome

After running **deployment_example.ipynb**, you’ll have a fully configured endpoint with:
- Correct model artifact loaded and validated
- Lineage-rich tags for audit/rollback
- Safe rollout strategy + autoscaling + alarms

After completing **deployment_exercise.ipynb**, you’ll have the same—**customized** to your dataset, model family, and operational constraints.
