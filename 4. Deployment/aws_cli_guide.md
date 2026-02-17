# AWS CLI Quick Guide

## Install

```bash
# macOS
brew install awscli

# Or with conda
conda install -c conda-forge awscli
```

Verify: `aws --version`

---

## SSO Login (Our Setup)

We use AWS SSO (Single Sign-On) to authenticate. No passwords or access keys needed.

### Available Profiles

All profiles connect to account `733246370304` via SSO URL `https://d-c6671a85bf.awsapps.com/start`.

> **Role** = your permissions in AWS. **Profile** = the name you use in the CLI (`--profile`).

| Profile | Role |
|---------|------|
| `DataScientist-733246370304` | DataScientist |
| `SageMaker-Full-AI-Access-733246370304` | SageMaker-Full-AI-Access |
| `AdministratorAccess-733246370304` | AdministratorAccess |

### First time setup

```bash
aws configure sso
```

It will ask:
- **SSO session name:** pick a name (e.g., `learningpods`)
- **SSO start URL:** `https://d-c6671a85bf.awsapps.com/start`
- **SSO region:** `af-south-1`
- **Account:** `733246370304`
- **Role:** Select your role from the list above
- **Profile name:** Use the format from the table above (e.g., `DataScientist-733246370304`)

### Login (do this every day)

```bash
# Students
aws sso login --profile DataScientist-733246370304

# Instructors
aws sso login --profile SageMaker-Full-AI-Access-733246370304
```

This opens your browser. Click "Allow" and you're in. The token lasts ~8 hours.

### "Token has expired" error?

Just run the login command again:
```bash
aws sso login --profile DataScientist-733246370304
```

---

## Common Commands

### Check who you are
```bash
aws sts get-caller-identity --profile SageMaker-Full-AI-Access-733246370304
```

### List S3 files
```bash
aws s3 ls s3://your-bucket/your-path/ --profile SageMaker-Full-AI-Access-733246370304
```

### List SageMaker pipelines
```bash
aws sagemaker list-pipelines --profile SageMaker-Full-AI-Access-733246370304 --region af-south-1
```

### Start a pipeline execution
```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name learningpods-pipeline \
  --profile SageMaker-Full-AI-Access-733246370304 \
  --region af-south-1
```

### Check pipeline execution status
```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name learningpods-pipeline \
  --profile SageMaker-Full-AI-Access-733246370304 \
  --region af-south-1
```

---

## Execution Role vs SSO Profile

These are two different things:

- **SSO Profile** — authenticates **you** (the human) to run CLI commands from your laptop.
- **Execution Role** — authenticates **SageMaker** (the service) to access AWS resources when running your pipeline in the cloud.

Our execution role: `SageMaker-ExecutionRole-20250915T083600`

You don't need to log in with the execution role — SageMaker assumes it automatically. It's configured in `pipeline.py` and has permissions to access S3, Redshift, ECR, and the model registry.

---

## Using Profiles in Python

The `--profile` flag is for the CLI. In Python (boto3), you pass it like this:

```python
import boto3

# Use a specific profile
session = boto3.Session(profile_name="SageMaker-Full-AI-Access-733246370304")
s3 = session.client('s3')
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Login | `aws sso login --profile YOUR_PROFILE` |
| Check identity | `aws sts get-caller-identity --profile YOUR_PROFILE` |
| List S3 files | `aws s3 ls s3://bucket/path/ --profile YOUR_PROFILE` |
| Copy to S3 | `aws s3 cp file.txt s3://bucket/path/ --profile YOUR_PROFILE` |
| List pipelines | `aws sagemaker list-pipelines --profile YOUR_PROFILE --region af-south-1` |
| Run pipeline | `aws sagemaker start-pipeline-execution --pipeline-name NAME --profile YOUR_PROFILE --region af-south-1` |
