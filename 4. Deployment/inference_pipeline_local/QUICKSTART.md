# Quick Start

```bash
pip install -r requirements.txt
```

## Using the model from SageMaker (recommended)

```bash
python infer.py --from-registry --input data.parquet --output predictions.csv
```

## Using a local model

```bash
python infer.py --model-dir ../training_pipeline_simplified/models --input data.parquet
```

**See README.md for full documentation.**
