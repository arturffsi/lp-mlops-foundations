"""Shared dataset configuration for Module 2 — Training.

All three example notebooks import from here so dataset-specific column names
live in exactly one place. Edit this file if you adapt the course to a
different table.
"""

# --- Target ---
TARGET_COL = "churn"

# --- Feature-type lists (used by 1_data_preparation_example.ipynb) ---
# Columns where missing values are filled with -1 and cast to int.
INT_FEATURES = [
    "codigocontaservico",
    "codigocliente",
    "n_dias_subscricao",
]

# Columns dropped by clean_data() — models can't consume raw datetimes.
DATE_FEATURES = [
    "iddim_date_inicio",
    "iddim_date_fim",
]

# --- ID columns to drop before training (used by notebooks 2 and 3) ---
# These uniquely identify a row and would let the model "cheat" if kept.
ID_COLS = [
    "idconsumo",
    "codigocontaservico",
    "idconta",
    "iddim_cliente",
    "idcliente",
    "codigocliente",
    "iddim_conta",
    "codigoconta",
]

# --- Business target ---
# Fraction of true churners we want to catch. Threshold is tuned to hit this.
TARGET_RECALL = 0.80

# --- Reproducibility ---
SEED = 42
