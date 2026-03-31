---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.1rem;
  }
  h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 0.2em; }
  h2 { color: #16213e; }
  h3 { color: #e94560; }
  code { background: #f4f4f4; padding: 0.1em 0.4em; border-radius: 4px; }
  pre { font-size: 0.8rem; }
  table { font-size: 0.85rem; }
---

# Module 5: CI/CD for ML Pipelines

### From manual runs to automated deployments

---

# What is CI/CD?

**CI/CD** stands for **Continuous Integration / Continuous Delivery**.

It's the practice of automatically testing and deploying your code every time you push a change — instead of doing it manually.

| Term | What it means |
|------|--------------|
| **Continuous Integration** | Every code change is automatically validated |
| **Continuous Delivery** | Every validated change is automatically deployed |

> Think of it as a robot that watches your GitHub repo and runs commands for you whenever you push code.

---

# Why does this matter for ML?

Without CI/CD, this is what happens every time you change the training code:

```
1. You make a change to train.py on your laptop
2. You remember to run: python pipeline.py --create
3. You remember to run: python pipeline.py --execute
4. You hope your teammate does the same...
5. Nobody knows which version of the code is running in AWS
```

**The problems:**
- Human error — someone forgets to run `--create` after a code change
- No audit trail — who deployed what, and when?
- Not scalable — doesn't work for a team

---

# With CI/CD

```
1. You push code to GitHub
2. GitHub automatically runs pipeline.py --create
3. GitHub automatically runs pipeline.py --execute
4. The whole team can see what ran and when
```

The code in SageMaker is **always in sync** with what's in GitHub.

---

# The pieces involved

| Component | Role | Analogy |
|-----------|------|---------|
| **GitHub** | Stores code, tracks every change | Google Docs version history |
| **GitHub Actions** | Runs commands when you push | A robot watching your repo |
| **`pipeline.py --create`** | Uploads code to SageMaker | Deploying the blueprint |
| **`pipeline.py --execute`** | Starts a training run | Pressing the "Run" button |
| **SageMaker** | Does the actual training on AWS | The factory floor |

None of these know about each other by default — **CI/CD is the glue**.

---

# What is GitHub Actions?

GitHub Actions is a tool built into GitHub that lets you run commands automatically in response to events (like a push).

You describe what to do in a **workflow file** — a YAML file inside `.github/workflows/`.

```
your-repo/
├── .github/
│   └── workflows/
│       └── update-training-pipeline.yml   ← this is the CI/CD config
├── 4. Deployment/
│   └── training_pipeline_simplified/
│       ├── train.py
│       └── pipeline.py
```

When you push to `main`, GitHub reads the workflow file and executes it on a fresh machine in the cloud.

---

# Our workflow file

```yaml
name: Update & Execute Training Pipeline

on:
  push:
    branches: [main]
    paths:
      - '4. Deployment/training_pipeline_simplified/**'
      - '!4. Deployment/training_pipeline_simplified/*.md'
```

**`on: push`** — triggers when code is pushed to GitHub

**`branches: [main]`** — only on the main branch (not feature branches)

**`paths`** — only when pipeline files change, not README edits

> This means: pushing a fix to `train.py` triggers the workflow. Pushing a change to a markdown file does not.

---

# What the workflow does — step by step

```yaml
steps:
  - uses: actions/checkout@v4               # 1. Download our code onto the runner

  - uses: actions/setup-python@v5           # 2. Install Python 3.12
    with:
      python-version: '3.12'

  - name: Install dependencies              # 3. pip install our packages
    run: pip install sagemaker boto3 pyyaml catboost pyarrow scikit-learn

  - uses: aws-actions/configure-aws-credentials@v4  # 4. Connect to AWS
    with:
      aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
      aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      aws-region: af-south-1

  - name: Update pipeline definition        # 5. Upload new code to SageMaker
    run: python pipeline.py --create

  - name: Execute pipeline                  # 6. Start a training run
    run: python pipeline.py --execute
```

---

# The runner — what is it?

When the workflow triggers, GitHub spins up a **fresh Ubuntu machine** (called a "runner") just for this job.

```
Your laptop          GitHub              Runner (Ubuntu)
──────────           ──────              ───────────────
git push    ──→      detects push  ──→   fresh machine boots
                                         installs Python
                                         installs packages
                                         connects to AWS
                                         runs pipeline.py --create
                                         runs pipeline.py --execute
                                         machine is destroyed
```

It's like a clean laptop that boots, does the job, and disappears — every time.

---

# AWS credentials — how does the runner connect to AWS?

The runner has no AWS credentials by default. We need to give them securely.

**Never put credentials directly in the YAML file** — it's public code!

Instead, store them as **GitHub Secrets**:

1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Add:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

GitHub encrypts them and injects them at runtime:
```yaml
aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
```

> Secrets are never shown in logs — even if someone reads the workflow, they can't see the values.

---

# Setup checklist — what you need before it works

**1. Create an IAM user for GitHub Actions** (AWS Console → IAM → Users → Create user)
- Name it `github-actions-sagemaker`
- Select **"Attach policies directly"**
- Attach: `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`
- Also needs `iam:PassRole` on the SageMaker execution role
- ⚠️ If you get "Access denied to iam:ListPolicies" — your SSO role doesn't have IAM permissions. Ask your AWS admin to attach the policies for you.

**2. Generate an access key for that user**
- IAM → Users → `github-actions-sagemaker` → **Security credentials** → **Create access key**
- Select "CI/CD" as the use case
- Copy the `Access Key ID` and `Secret Access Key` — shown only once

**3. Add the keys as GitHub Secrets**
- GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
- Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

**4. Test it**
- Push any change to a file under `4. Deployment/training_pipeline_simplified/`
- Go to GitHub → **Actions** tab → watch the workflow run

---

# How CI/CD connects to our SageMaker pipeline

In Module 4 you ran the pipeline manually from your terminal:

```
python pipeline.py --create    # upload code to SageMaker
python pipeline.py --execute   # start a training run
```

CI/CD just automates those exact same two commands — nothing changes in SageMaker itself.

```
                    GitHub Actions
                   ┌──────────────────────────────────┐
git push ──────→   │  python pipeline.py --create      │
                   │  python pipeline.py --execute      │
                   └──────────────┬───────────────────-┘
                                  ↓
                         SageMaker Pipeline
                   ┌──────────────────────────────────┐
                   │  Step 1: ExportFromRedshift        │
                   │  Step 2: TrainChurnModel           │
                   │  Step 3: EvaluateModel             │
                   │  Step 4: CheckModelQuality         │
                   └──────────────────────────────────┘
```

The SageMaker pipeline is unchanged — CI/CD is just the trigger.

---

# Manual vs CI/CD — side by side

### Manual (what we've been doing)
```
You change train.py
    → you run: python pipeline.py --create
    → you run: python pipeline.py --execute
    → you check the SageMaker console
```

### CI/CD (automated)
```
You change train.py
    → git add, git commit, git push
    → GitHub Actions runs --create and --execute automatically
    → team sees the run in GitHub Actions tab
```

The only difference is `git push` replaces the manual steps.

---

# Common CI/CD patterns

### Pattern 1: Update only (good for learning)
Push updates the pipeline definition. You still run `--execute` manually.
```
git push → --create
```

### Pattern 2: Update + Execute (automated retraining)
Every push to main triggers a full training run.
```
git push → --create --execute
```

### Pattern 3: Scheduled retraining
Retrain on a fixed schedule, regardless of code changes.
```yaml
on:
  schedule:
    - cron: '0 8 * * 1'  # Every Monday at 8 AM UTC
```

### Pattern 4: Pull request validation
Validate the pipeline definition before merging (no execute, just create).
```
open PR → --create (validates definition) → merge
```

---

# What happens if the pipeline fails?

GitHub Actions shows a red cross on the commit. The team can see exactly which step failed and read the logs.

```
✅ Checkout code
✅ Install Python
✅ Install dependencies
✅ Configure AWS credentials
✅ Update pipeline definition
❌ Execute pipeline          ← failed here, click to see logs
```

This is much better than "I ran it on my laptop and it worked" — everything is visible, logged, and reproducible.

---

# Summary

| Question | Answer |
|----------|--------|
| What is CI/CD? | Automatically run commands when you push code |
| What tool do we use? | GitHub Actions |
| When does it trigger? | On push to `main` when pipeline files change |
| What does it run? | `pipeline.py --create` then `pipeline.py --execute` |
| How does it connect to AWS? | GitHub Secrets (encrypted credentials) |
| Does pushing always retrain? | Only if pipeline files changed (path filter) |
| What if it fails? | Red cross on the commit, full logs visible |

---

# What's next: Module 6 — Monitoring

The model is now trained and deployed automatically. But how do we know it's still performing well in production?

**Module 6 covers:**
- Data drift — input data changing over time
- Model drift — predictions degrading
- Alerts and retraining triggers
- Keeping models reliable after deployment

> CI/CD gets the model to production. Monitoring keeps it healthy once it's there.
