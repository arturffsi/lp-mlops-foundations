# CI/CD for the Training Pipeline

## How `--create` Works

When you run:

```bash
python pipeline.py --create
```

This does the following:

1. **Packages your code** — `train.py`, `train_utils.py`, `data_loader.py`, `config.yaml`, `export_redshift.py`, and `evaluate_model.py` get uploaded to S3
2. **Uploads the pipeline definition** — The DAG (steps, parameters, conditions) is sent to SageMaker
3. **Overwrites the previous definition** — There's no versioning; the latest `--create` always wins

After `--create`, the pipeline exists in SageMaker but **nothing runs yet**. You need `--execute` to actually start a training run.

### When do you need to re-run `--create`?

| Change | Need `--create`? | Why |
|--------|:---:|-----|
| Changed `train.py` or `train_utils.py` | Yes | Code is re-packaged and uploaded to S3 |
| Changed `config.yaml` | Yes | Config is bundled with the code |
| Changed hyperparameter defaults in `pipeline.py` | Yes | Pipeline definition changes |
| Changed quality gate thresholds | Yes | Condition step changes |
| Just want to re-run with same code | No | Just use `--execute` |
| Override hyperparameters at runtime | No | Pass them as `--execute` parameters |

---

## Manual vs Automated Workflow

### Current Workflow (Manual)

```
Developer makes code changes
        ↓
Developer runs: python pipeline.py --create
        ↓
Developer runs: python pipeline.py --execute
        ↓
Developer monitors in SageMaker Console
```

This works fine for experimentation, but in a team setting you want changes to be deployed automatically when code is merged.

### CI/CD Workflow (Automated)

```
Developer pushes code to GitHub
        ↓
GitHub Actions detects changes in pipeline files
        ↓
Automatically runs: python pipeline.py --create
        ↓
Optionally runs: python pipeline.py --execute
        ↓
Team gets notified of results
```

---

## Does the Pipeline Run When You Push?

**No, not by default.** SageMaker Pipelines don't know about your Git repository. You need a CI/CD tool (like GitHub Actions) to connect the two.

Here's what each piece does:

| Component | Role |
|-----------|------|
| **Git/GitHub** | Stores your code, tracks changes |
| **GitHub Actions** | Runs commands when you push code |
| **`pipeline.py --create`** | Updates the pipeline definition in SageMaker |
| **`pipeline.py --execute`** | Starts a training run |
| **SageMaker** | Runs the actual training on AWS |

Without CI/CD, pushing code to GitHub does nothing to your SageMaker pipeline.

---

## GitHub Actions Example

Here's a simple workflow that updates the pipeline whenever you push changes:

```yaml
# .github/workflows/update-pipeline.yml

name: Update Training Pipeline

# Only trigger when pipeline code changes on main branch
on:
  push:
    branches: [main]
    paths:
      - '4. Deployment/training_pipeline_simplified/**'

jobs:
  update-pipeline:
    runs-on: ubuntu-latest

    steps:
      # 1. Get the code
      - uses: actions/checkout@v4

      # 2. Set up Python
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      # 3. Install dependencies
      - run: |
          pip install sagemaker boto3 pyyaml catboost pyarrow scikit-learn

      # 4. Configure AWS credentials (from GitHub Secrets)
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: af-south-1

      # 5. Update the pipeline definition
      - name: Update Pipeline
        working-directory: '4. Deployment/training_pipeline_simplified'
        run: python pipeline.py --create

      # 6. (Optional) Execute the pipeline
      # Uncomment the next lines if you want to auto-run on every push
      # - name: Execute Pipeline
      #   working-directory: '4. Deployment/training_pipeline_simplified'
      #   run: python pipeline.py --execute
```

### What This Does

1. **Triggers only when relevant files change** — Pushing changes to unrelated files won't trigger the pipeline
2. **Updates the pipeline definition** — Ensures SageMaker always has the latest version of your code
3. **Execution is optional** — You can uncomment the execute step if you want every push to trigger training

---

## Setting Up AWS Credentials in GitHub

For GitHub Actions to access your AWS account, you need to store credentials as secrets:

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Add these secrets:
   - `AWS_ACCESS_KEY_ID` — Your AWS access key
   - `AWS_SECRET_ACCESS_KEY` — Your AWS secret key
3. These are encrypted and never shown in logs

> **Better approach for production:** Use OIDC (OpenID Connect) instead of long-lived access keys. This lets GitHub Actions assume an IAM role directly without storing credentials. See [GitHub docs on OIDC with AWS](https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services).

---

## Common CI/CD Patterns

### Pattern 1: Update Only (Recommended for Learning)
Push updates the pipeline definition, but you manually execute from the console or CLI.

```
git push → GitHub Actions → pipeline.py --create
```

### Pattern 2: Update + Execute
Every push to main triggers a full training run. Good for automated retraining.

```
git push → GitHub Actions → pipeline.py --create --execute
```

### Pattern 3: Scheduled Retraining
Use a cron schedule to retrain on a fixed cadence (e.g., weekly).

```yaml
on:
  schedule:
    - cron: '0 8 * * 1'  # Every Monday at 8 AM UTC
```

### Pattern 4: Pull Request Validation
Run `--create` (without `--execute`) on PRs to verify the pipeline definition is valid before merging.

```yaml
on:
  pull_request:
    paths:
      - '4. Deployment/training_pipeline_simplified/**'
```

---

## Summary

| Question | Answer |
|----------|--------|
| Does the pipeline run when I push? | No, not without CI/CD |
| What does `--create` do? | Uploads code + pipeline definition to SageMaker |
| What does `--execute` do? | Starts a training run using the latest definition |
| Do I need `--create` every time? | Only when code or pipeline config changes |
| What CI/CD tool should I use? | GitHub Actions (simplest for GitHub repos) |
