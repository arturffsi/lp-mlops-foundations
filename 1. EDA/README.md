# Module 1 — Exploratory Data Analysis (EDA)

> **Where you are in the course:** Week 1 of 6. You're starting from raw data and you'll finish the week with a clean, versioned dataset ready for training in Module 2.

## What you'll learn

By the end of this module you will be able to:

1. **Load data reproducibly** from either a local Parquet file (fast) or Amazon Redshift via `UNLOAD` (production pattern).
2. **Run the 5 data-quality checks** every MLOps pipeline needs: completeness, consistency, accuracy, validity, timeliness.
3. **Analyze the target variable** and recognize class imbalance.
4. **Explore features** (univariate and bivariate) and rank them by their correlation with the target.
5. **Split data without leakage** using a time-based train / validation / test split.
6. **Export MLOps artifacts** (`feature_schema.json`, `data_metadata.json`, versioned splits) that feed directly into Module 2.

## Files in this folder

| File | What it is | When to use it |
|------|------------|----------------|
| `eda_example.ipynb` | **Worked example** — full EDA on the churn dataset. Start here. | Read top to bottom. |
| `eda_exercise.ipynb` | **Hands-on exercises** with `# YOUR CODE HERE` blanks. | After reading the example. |
| `load_redshift.ipynb` | Minimal example of pulling a table from Redshift to local. | Reference for the Redshift connection. |
| `data_io.py` | `load_data()` helper used by all notebooks. | You import it; you don't edit it. |
| `eda_example_results/` | Output folder created when you run the notebook. | Inspect outputs here. |

## Recommended path for beginners

1. **Read the intro** in the first cell of `eda_example.ipynb` — it explains *why* EDA matters in an MLOps pipeline.
2. **Run `eda_example.ipynb` top to bottom** using the bundled local sample (`../sample_data_from_redshift/sample.parquet`). No AWS account needed.
3. **Open `eda_exercise.ipynb`** and fill in the `# YOUR CODE HERE` blanks on your own.
4. **(Optional)** Re-run the example against Redshift to see the production data path.

## Quick start (local, no AWS needed)

```bash
cd "1. EDA"
pip install -r requirements.txt
jupyter notebook eda_example.ipynb
```

The pinned versions in `requirements.txt` are the ones the notebook is known to run on — newer pandas/pyarrow have compatibility bugs that break `read_parquet`. The first data-loading cell defaults to the local sample file, so you can run the whole notebook with no further setup.

## Running against Redshift (optional)

> **First time on AWS?** You'll need an SSO session before the notebook can talk to Redshift. See [../prerequisites/aws_cli_guide.md](../prerequisites/aws_cli_guide.md) for the one-time setup and the daily `aws sso login` command. Without valid credentials, boto3 will raise `NoCredentialsError`.

The notebook's data-loading cell has a `DATA_SOURCE = "redshift"` switch. Flip it to `"redshift"` and set `USER_NAME` to your own identifier (so your S3 export folder doesn't collide with anyone else's).

The `REDSHIFT_CONFIG` dictionary must match the keys expected by `data_io.load_data()`:

```python
REDSHIFT_CONFIG = {
    'cluster_id':        'redshift-cluster-dsi',
    'database':          'prod',
    'db_user':           'svc_sagemaker',
    'region':            'af-south-1',
    'iam_role':          'arn:aws:iam::733246370304:role/RedshiftIAMAuthRole',
    's3_export_prefix':  f's3://sagemaker-af-south-1-733246370304/redshift_exports/{USER_NAME}/eda_train_{date.today():%Y-%m-%d}',
    'cleanup_s3':        False,   # True to delete the temp S3 files after loading
}

df = load_data(
    source='redshift',
    sql='SELECT * FROM dth_churn_ml_training.training_features WHERE RANDOM() < 0.0001;',
    redshift_kwargs=REDSHIFT_CONFIG,
)
```

Under the hood `load_data()` runs an `UNLOAD` to S3 as Parquet — the same pattern the production pipeline uses, so what you see in EDA is what runs in prod.

## Security notes

- The example credentials in the notebook are placeholders. Never commit real credentials to git.
- We use IAM authentication (no password). In production, prefer environment variables or AWS Secrets Manager.
- Temporary S3 exports can be auto-cleaned with `cleanup_s3=True`.

## What this module produces

After a full run of `eda_example.ipynb` you should find:

```
eda_example_results/
├── train.parquet              # ~80% of rows, earliest dates
├── val.parquet                # ~10% of rows, middle dates
├── test.parquet               # ~10% of rows, latest dates
├── feature_schema.json        # column types, ranges, categories
└── data_metadata.json         # source, extraction date, row counts, target distribution
```

These artifacts are **the input to Module 2 (Training)** — versioned, leakage-safe, and traceable.

## Peer-validation checklist

Before submitting, your work should satisfy:

- [ ] Data loading is reproducible (seeded, or query is committed)
- [ ] Each feature in `feature_schema.json` has a documented type and rationale
- [ ] Train / val / test split is time-based and the date ranges don't overlap
- [ ] Both `data_metadata.json` and the split files are present in `eda_example_results/`
