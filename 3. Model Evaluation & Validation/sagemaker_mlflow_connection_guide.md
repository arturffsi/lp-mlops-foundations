# 🔧 Connecting to a SageMaker MLflow Tracking Server from a Local Machine

This guide explains how to connect and log experiments to an **AWS SageMaker MLflow Tracking Server** from your **local environment or Jupyter notebook**, using **AWS SSO** for authentication.

---

## ✅ Prerequisites

- AWS CLI installed (`aws --version`)
- Python ≥3.9 with `pip`
- Access to your company’s AWS SSO portal  
  Example: `https://d-c6671a85bf.awsapps.com/start`
- SageMaker MLflow Tracking Server ARN  
  ```
  arn:aws:sagemaker:af-south-1:733246370304:mlflow-tracking-server/dth-churn-ml
  ```
- For Windows:
  - [aws.amazon.com](https://aws.amazon.com/cli/)
---

## 🧭 Step 1 — Configure AWS SSO locally

```bash
aws configure sso
```

Fill in when prompted:

```
SSO start URL [None]: https://d-c6671a85bf.awsapps.com/start
SSO region [None]: af-south-1
SSO account ID [None]: 733246370304
SSO role name [None]: AdministratorAccess
CLI default client Region [None]: af-south-1
CLI default output format [None]: json
```

Then log in:

```bash
aws sso login --profile your-profile
```

Verify:
```bash
aws sts get-caller-identity --profile your-profile
```

You should see your AWS account and user ARN.

---

## ⚙️ Step 2 — Install dependencies

```bash
pip install "mlflow==3.0.0" sagemaker-mlflow
```

Make sure versions match your SageMaker MLflow server (`3.0.0` in this case).

---

## 🧩 Step 3 — Environment setup

Export these variables in your terminal or `.env` file:

```bash
export AWS_PROFILE=your-profile
export AWS_REGION=af-south-1
export MLFLOW_TRACKING_URI=arn:aws:sagemaker:af-south-1:733246370304:mlflow-tracking-server/dth-churn-ml
export MLFLOW_REGISTRY_URI=sqlite:///:memory:
```

---

## 🧪 Step 4 — Test logging (Python script)

```python
import sagemaker_mlflow  # registers the 'arn:' scheme
import mlflow

mlflow.set_experiment("local-connection-test")

with mlflow.start_run(run_name="from_local"):
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", 0.523)
    with open("hello.txt", "w") as f:
        f.write("hello from local")
    mlflow.log_artifact("hello.txt")

print("✅ Logged run + artifact to SageMaker MLflow")
```

Output:
```
🏃 View run from_local at: https://af-south-1.experiments.sagemaker.aws/#/experiments/1/runs/<run-id>
✅ Logged run + artifact to SageMaker MLflow
```

---

## 📓 Step 5 — Use in Jupyter Notebooks

Place this cell **at the top** of your notebook before using MLflow:

```python
import os, sagemaker_mlflow, mlflow

os.environ["AWS_PROFILE"] = "your-profile"
os.environ["AWS_REGION"] = "af-south-1"
os.environ["MLFLOW_TRACKING_URI"] = "arn:aws:sagemaker:af-south-1:733246370304:mlflow-tracking-server/dth-churn-ml"
os.environ["MLFLOW_REGISTRY_URI"] = "sqlite:///:memory:"

print("Tracking URI ->", mlflow.get_tracking_uri())
print("Registry URI  ->", mlflow.get_registry_uri())
```

Then use MLflow as usual:
```python
mlflow.set_experiment("local-connection-test")
with mlflow.start_run(run_name="notebook-test"):
    mlflow.log_param("lr", 0.001)
    mlflow.log_metric("accuracy", 0.93)
```

---

## 🧭 Step 6 — View experiments

Open your runs in the SageMaker MLflow console:  
👉 [https://af-south-1.experiments.sagemaker.aws](https://af-south-1.experiments.sagemaker.aws)

---

## 🧠 Common issues

| Error | Cause | Fix |
|-------|--------|-----|
| `Missing Tracking Server ARN` | Using HTTPS URL instead of ARN | Use the full ARN as `MLFLOW_TRACKING_URI` |
| `KeyError: 'arn'` | Registry URI still set to ARN | Set `MLFLOW_REGISTRY_URI=sqlite:///:memory:` |
| `AccessDenied` | Expired SSO session | Run `aws sso login --profile your-profile` again |
| `SignatureDoesNotMatch` | Wrong region | Ensure `AWS_REGION=af-south-1` |

---

### ✅ You can now log experiments locally, and all runs will appear in your SageMaker MLflow tracking server.
