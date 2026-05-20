"""
Build inference features locally from Redshift raw tables.

Replicates the inference SQL pipeline in Python/Pandas:
  1. Sample random codigocontaservico from yesterday's parque snapshot
  2. Build contract sequence features (tenure, gaps, stats)
  3. Join with client/account dimensions
  4. Add campaign features (30-day window)
  5. Add topup features (90-day window)
  6. Add past_churns from churner lookup

Usage:
  python build_inference_features.py                    # Download + build
  python build_inference_features.py --skip-download    # Rebuild from cached samples
  python build_inference_features.py --sample-size 1000 # Fewer service codes
"""

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg2

# Redshift connection (dev database — raw tables are here and updated daily)
REDSHIFT_HOST = "redshift-cluster-dsi.cl4o4mmtx9ir.af-south-1.redshift.amazonaws.com"
REDSHIFT_PORT = 5439
REDSHIFT_DB = "dev"
REDSHIFT_USER = "awsuser"

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
OUTPUT_FILE = os.path.join(SAMPLE_DIR, "inference_features.parquet")

# Cache files for --skip-download
CACHE_FILES = {
    "parque": os.path.join(SAMPLE_DIR, "raw_parque.parquet"),
    "parque_history": os.path.join(SAMPLE_DIR, "raw_parque_history.parquet"),
    "cliente": os.path.join(SAMPLE_DIR, "raw_cliente_unified.parquet"),
    "campaigns": os.path.join(SAMPLE_DIR, "raw_campaigns.parquet"),
    "topups": os.path.join(SAMPLE_DIR, "raw_topups.parquet"),
    "churners": os.path.join(SAMPLE_DIR, "raw_churners.parquet"),
}


def get_password():
    """Load Redshift password from .env file."""
    env_path = os.path.join(
        os.path.dirname(__file__), "..", "training_pipeline_simplified", ".env"
    )
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("REDSHIFT_PASSWORD="):
                    return line.split("=", 1)[1]
    return os.getenv("REDSHIFT_PASSWORD")


def connect():
    """Connect to Redshift dev database."""
    return psycopg2.connect(
        host=REDSHIFT_HOST,
        port=REDSHIFT_PORT,
        dbname=REDSHIFT_DB,
        user=REDSHIFT_USER,
        password=get_password(),
        connect_timeout=15,
    )


def download_raw_data(sample_size=500, reference_date=None):
    """Download raw data samples from Redshift for a set of random service codes."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print("Connecting to Redshift (dev)...")
    conn = connect()
    cursor = conn.cursor()

    # Determine reference date (latest available parque date)
    if reference_date is None:
        cursor.execute("SELECT MAX(iddim_data_parque) FROM staging_ao.stg_parque_dth_ativo_raw;")
        ref_int = cursor.fetchone()[0]
        reference_date = pd.to_datetime(str(ref_int), format="%Y%m%d").date()
    ref_int = int(reference_date.strftime("%Y%m%d"))
    print(f"Reference date: {reference_date} ({ref_int})")

    # 1. Sample random codigocontaservico from reference date
    print(f"\n[1/6] Sampling {sample_size} random service codes from parque ({reference_date})...")
    cursor.execute(f"""
        SELECT DISTINCT codigocontaservico
        FROM staging_ao.stg_parque_dth_ativo_raw
        WHERE iddim_data_parque = {ref_int}
        ORDER BY RANDOM()
        LIMIT {sample_size};
    """)
    codigos = [r[0] for r in cursor.fetchall()]
    print(f"  Got {len(codigos)} service codes")
    codigos_str = ",".join(f"'{c}'" for c in codigos)

    # 2. Download parque data for these codes (reference date snapshot)
    print(f"\n[2/6] Downloading parque snapshot for reference date...")
    parque_sql = f"""
        SELECT idconsumo, id_contaservico, codigocontaservico, idconta,
               iddim_data_parque, iddim_date_inicio, iddim_date_fim,
               id_produto_actual, tipo_produto_actual, tipo_subscricao, tipo_stb,
               n_dias_subscricao, n_dias_expirado
        FROM staging_ao.stg_parque_dth_ativo_raw
        WHERE iddim_data_parque = {ref_int}
          AND codigocontaservico IN ({codigos_str});
    """
    df_parque = pd.read_sql(parque_sql, conn)
    print(f"  Got {len(df_parque):,} rows")

    # 3. Download full parque history for contract sequence (last 18 months)
    # Only expiry-day snapshots (iddim_data_parque = iddim_date_fim) to get one row per contract
    # This matches training: WHERE iddim_data_parque = iddim_date_fim
    cutoff_int = int((reference_date - timedelta(days=548)).strftime("%Y%m%d"))
    print(f"\n[3/6] Downloading parque history ({cutoff_int} to {ref_int}, expiry-day only)...")
    history_sql = f"""
        SELECT DISTINCT idconsumo, codigocontaservico, iddim_date_inicio, iddim_date_fim
        FROM staging_ao.stg_parque_dth_ativo_raw
        WHERE codigocontaservico IN ({codigos_str})
          AND iddim_data_parque = iddim_date_fim
          AND iddim_data_parque >= {cutoff_int}
          AND iddim_date_inicio IS NOT NULL
          AND iddim_date_fim IS NOT NULL;
    """
    df_history = pd.read_sql(history_sql, conn)
    print(f"  Got {len(df_history):,} rows")

    # 4. Download client/account dimensions
    # First get the idconta values to join through
    idconta_list = df_parque["idconta"].dropna().unique().tolist()
    idconta_str = ",".join(str(int(c)) for c in idconta_list)
    print(f"\n[4/6] Downloading client/account dimensions...")

    cliente_sql = f"""
        SELECT
            cl.iddim_cliente, cl.idcliente, cl.codigocliente,
            cl.provincia, cl.municipio, cl.data_nascimento, cl.sexo, cl.estado_civil,
            cl.data_registo, cl.data_alteracao, cl.version AS cl_version,
            co.iddim_conta, co.idconta, co.codigoconta,
            co.idgrupo_dim_contadimensao, co.tipoconta, co.tecnologia,
            co.iddim_data_criacao AS conta_data_criacao, co.versao AS co_version,
            cs.iddim_contaservico_dth, cs.idcontaservico,
            cs.codigo_contaservico, cs.iddim_data_criacao AS cs_data_criacao,
            cs.versao AS cs_version
        FROM dwao.dim_conta co
        JOIN dwao.dim_cliente cl ON cl.iddim_cliente = co.iddim_cliente
        JOIN dwao.dim_contaservico_dth_v2 cs ON cs.iddim_conta = co.iddim_conta
        WHERE co.idconta IN ({idconta_str})
          AND cs.codigo_contaservico IN ({codigos_str});
    """
    df_cliente = pd.read_sql(cliente_sql, conn)
    print(f"  Got {len(df_cliente):,} rows")

    # 5. Download campaign contacts (30-day window)
    window_start = reference_date - timedelta(days=30)
    print(f"\n[5/6] Downloading campaign contacts ({window_start} to {reference_date})...")
    campaign_sql = f"""
        SELECT codigo_contaservico AS codigocontaservico,
               CASE WHEN UPPER(TRIM(Sucesso)) = 'SIM' THEN 1 ELSE 0 END AS sucesso_flag,
               data_estado::DATE + COALESCE(hora::TIME, '00:00:00'::TIME) AS contact_ts
        FROM dwao.reporting_campanhas_ao_sucesso
        WHERE codigo_contaservico IN ({codigos_str})
          AND data_estado::DATE >= '{window_start}'
          AND data_estado::DATE < '{reference_date}';
    """
    df_campaigns = pd.read_sql(campaign_sql, conn)
    print(f"  Got {len(df_campaigns):,} campaign contacts")

    # 6. Download topups (90-day window) — need iddim_cliente first
    # Deduplicate cliente to get latest version per codigocliente
    df_cl_dedup = df_cliente.sort_values("cl_version").drop_duplicates(
        subset="codigocliente", keep="last"
    )
    cliente_ids = df_cl_dedup["iddim_cliente"].dropna().unique().tolist()

    topup_start = reference_date - timedelta(days=90)
    topup_start_int = int(topup_start.strftime("%Y%m%d"))
    ref_int_topup = int(reference_date.strftime("%Y%m%d"))
    print(f"\n[6/6] Downloading topups for {len(cliente_ids)} customers ({topup_start} to {reference_date})...")

    chunk_size = 500
    all_topups = []
    for i in range(0, len(cliente_ids), chunk_size):
        chunk = cliente_ids[i : i + chunk_size]
        ids_str = ",".join(str(int(c)) for c in chunk)
        topup_sql = f"""
            SELECT iddim_cliente,
                   TO_TIMESTAMP(
                       TO_CHAR(iddim_date, '99999999') || ' ' || LPAD(iddim_time::TEXT, 6, '0'),
                       'YYYYMMDD HH24MISS'
                   ) AS date_time,
                   valorcarregado,
                   iddim_tipo_carregamento,
                   NULL::BIGINT AS iddim_canal,
                   iddim_utilizador
            FROM dwao.fact_carregamentos
            WHERE iddim_cliente IN ({ids_str})
              AND iddim_date >= {topup_start_int}
              AND iddim_date < {ref_int_topup}
              AND valorcarregado IS NOT NULL AND valorcarregado <> 0

            UNION ALL

            SELECT iddim_cliente,
                   TO_TIMESTAMP(
                       TO_CHAR(iddim_date, '99999999') || ' ' || LPAD(iddim_time::TEXT, 6, '0'),
                       'YYYYMMDD HH24MISS'
                   ) AS date_time,
                   valorcarregado,
                   NULL::BIGINT AS iddim_tipo_carregamento,
                   iddim_canal,
                   iddim_utilizador
            FROM dwao.fact_carregamentos_v2
            WHERE iddim_cliente IN ({ids_str})
              AND iddim_date >= {topup_start_int}
              AND iddim_date < {ref_int_topup}
              AND valorcarregado IS NOT NULL AND valorcarregado <> 0;
        """
        chunk_df = pd.read_sql(topup_sql, conn)
        all_topups.append(chunk_df)
        print(f"  Chunk {i // chunk_size + 1}: {len(chunk_df):,} rows")

    df_topups = pd.concat(all_topups, ignore_index=True) if all_topups else pd.DataFrame()
    print(f"  Total topup rows: {len(df_topups):,}")

    # Download churner IDs for past_churns
    print("\n  Checking churner status...")
    all_idconsumo = df_history["idconsumo"].unique().tolist()
    churner_ids = []
    for i in range(0, len(all_idconsumo), 1000):
        chunk = all_idconsumo[i : i + 1000]
        chunk_str = ",".join(str(int(c)) for c in chunk)
        cursor.execute(f"""
            SELECT DISTINCT idconsumo FROM staging_ao.parque_dth_desligado_raw_hist
            WHERE idconsumo IN ({chunk_str});
        """)
        churner_ids.extend([r[0] for r in cursor.fetchall()])
    df_churners = pd.DataFrame({"idconsumo": churner_ids})
    print(f"  Found {len(df_churners):,} churned contracts")

    conn.close()

    # Save cache files
    df_parque.to_parquet(CACHE_FILES["parque"], index=False)
    df_history.to_parquet(CACHE_FILES["parque_history"], index=False)
    df_cliente.to_parquet(CACHE_FILES["cliente"], index=False)
    df_campaigns.to_parquet(CACHE_FILES["campaigns"], index=False)
    df_topups.to_parquet(CACHE_FILES["topups"], index=False)
    df_churners.to_parquet(CACHE_FILES["churners"], index=False)
    print("\nAll raw data cached to sample_data/")

    return df_parque, df_history, df_cliente, df_campaigns, df_topups, df_churners, reference_date


def build_features(df_parque, df_history, df_cliente, df_campaigns, df_topups, df_churners, reference_date):
    """Build all inference features from raw data (mirrors the SQL pipeline)."""
    ref_ts = pd.Timestamp(reference_date)

    # ── Step 1: Parse dates in parque snapshot ──
    print("\n[Step 1] Preparing parque snapshot...")
    df = df_parque.copy()
    for col in ["iddim_date_inicio", "iddim_date_fim"]:
        df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")

    # ── Step 2: Contract sequence features ──
    print("[Step 2] Building contract sequence features...")
    hist = df_history.copy()
    for col in ["iddim_date_inicio", "iddim_date_fim"]:
        hist[col] = pd.to_datetime(hist[col].astype(str), format="%Y%m%d", errors="coerce")
    hist = hist.dropna(subset=["iddim_date_inicio", "iddim_date_fim"])
    # Match reference pipeline: dedup per idconsumo keeping latest iddim_date_fim
    hist = (
        hist.sort_values(["idconsumo", "iddim_date_fim", "iddim_date_inicio"], ascending=[True, False, False])
            .drop_duplicates(subset=["idconsumo"], keep="first")
    )
    hist = hist.sort_values(["codigocontaservico", "iddim_date_fim", "idconsumo"])

    # Contract number (0-indexed)
    hist["contract_number"] = hist.groupby("codigocontaservico").cumcount()
    # Contract length
    hist["contract_len_days"] = (hist["iddim_date_fim"] - hist["iddim_date_inicio"]).dt.days
    # Previous expiry for gap calculation
    hist["prev_expiry"] = hist.groupby("codigocontaservico")["iddim_date_fim"].shift(1)
    hist["gap_since_prev_expiry"] = (hist["iddim_date_inicio"] - hist["prev_expiry"]).dt.days

    # Running stats on prior contract lengths
    hist["mean_len_prev"] = (
        hist.groupby("codigocontaservico")["contract_len_days"]
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    hist["std_len_prev"] = (
        hist.groupby("codigocontaservico")["contract_len_days"]
        .apply(lambda x: x.shift(1).expanding().std())
        .reset_index(level=0, drop=True)
    )
    hist["n_prev_contracts"] = hist["contract_number"]

    # Calculate tenure as reference_date - first_ever_contract_start per codigocontaservico
    # This matches training: tenure = total time as customer, not just current contract
    first_start = hist.groupby("codigocontaservico")["iddim_date_inicio"].min().reset_index()
    first_start.columns = ["codigocontaservico", "first_contract_start"]

    # Join contract stats back to parque snapshot
    contract_stats = hist[["idconsumo", "contract_number", "contract_len_days",
                           "gap_since_prev_expiry", "mean_len_prev", "std_len_prev",
                           "n_prev_contracts"]].drop_duplicates(subset="idconsumo", keep="last")
    df = df.merge(contract_stats, on="idconsumo", how="left")
    df = df.merge(first_start, on="codigocontaservico", how="left")

    # Tenure = total time as customer (from first ever contract to reference date)
    df["tenure_days"] = (ref_ts - df["first_contract_start"]).dt.days
    df = df.drop(columns=["first_contract_start"])
    df["expiry_month"] = df["iddim_date_fim"].dt.month.astype("Int64")
    df["expiry_dow"] = df["iddim_date_fim"].dt.dayofweek.astype("Int64")
    df["tenure_bucket"] = pd.cut(
        df["tenure_days"].fillna(0),
        bins=[-1, 180, 365, float("inf")],
        labels=["<6 mo", "6–12 mo", ">1 yr"],
    ).astype(str)

    print(f"  {len(df):,} rows with contract features")

    # ── Step 3: Client/account dimensions ──
    print("[Step 3] Joining client/account dimensions...")
    cl = df_cliente.copy()

    # Match reference pipeline: dedup each dimension table independently, then join
    # 1. dim_cliente: latest version per codigocliente
    cl_dedup = (cl[["iddim_cliente", "idcliente", "codigocliente", "provincia", "municipio",
                     "data_nascimento", "sexo", "estado_civil", "data_registo", "data_alteracao", "cl_version"]]
                .drop_duplicates()
                .sort_values("cl_version")
                .drop_duplicates(subset="codigocliente", keep="last")
                .drop(columns=["cl_version"]))
    # 2. dim_conta: latest version per codigoconta
    co_dedup = (cl[["iddim_conta", "idconta", "codigoconta", "iddim_cliente",
                     "idgrupo_dim_contadimensao", "tipoconta", "tecnologia", "conta_data_criacao", "co_version"]]
                .drop_duplicates()
                .sort_values("co_version")
                .drop_duplicates(subset="codigoconta", keep="last")
                .drop(columns=["co_version"]))
    # 3. dim_contaservico: latest version per codigo_contaservico
    cs_dedup = (cl[["iddim_contaservico_dth", "idcontaservico", "codigo_contaservico",
                     "iddim_conta", "cs_data_criacao", "cs_version"]]
                .drop_duplicates()
                .sort_values("cs_version")
                .drop_duplicates(subset="codigo_contaservico", keep="last")
                .drop(columns=["cs_version"]))
    # Re-join them
    cl = cs_dedup.merge(co_dedup, on="iddim_conta", how="left")
    cl = cl.merge(cl_dedup, on="iddim_cliente", how="left")

    # Calculate age
    cl["data_nascimento"] = pd.to_datetime(cl["data_nascimento"], errors="coerce")
    cl["age"] = np.where(
        (cl["data_nascimento"].isna()) | (cl["data_nascimento"] < "1900-01-01") | (cl["data_nascimento"] > str(reference_date)),
        np.nan,
        (ref_ts - cl["data_nascimento"]).dt.days / 365.25,
    )
    cl["age"] = cl["age"].round(2)
    cl["age_missing"] = cl["age"].isna().astype(int)

    # Account ages
    cl["data_registo"] = pd.to_datetime(cl["data_registo"], errors="coerce")
    cl["data_alteracao"] = pd.to_datetime(cl["data_alteracao"], errors="coerce")
    cl["account_age_d_cliente"] = (ref_ts - cl["data_registo"]).dt.days
    cl["days_since_last_update_cliente"] = (ref_ts - cl["data_alteracao"]).dt.days

    # Conta/contaservico ages
    cl["conta_data_criacao"] = pd.to_datetime(cl["conta_data_criacao"].astype(str), format="%Y%m%d", errors="coerce")
    cl["cs_data_criacao"] = pd.to_datetime(cl["cs_data_criacao"].astype(str), format="%Y%m%d", errors="coerce")
    cl["account_age_d_conta"] = (ref_ts - cl["conta_data_criacao"]).dt.days
    cl["account_age_d_contaservico"] = (ref_ts - cl["cs_data_criacao"]).dt.days

    # Select columns for join
    cl_cols = cl[["codigo_contaservico", "iddim_cliente", "idcliente", "codigocliente",
                  "provincia", "municipio", "age", "age_missing", "sexo", "estado_civil",
                  "account_age_d_cliente", "days_since_last_update_cliente",
                  "iddim_conta", "codigoconta", "idgrupo_dim_contadimensao", "tipoconta",
                  "tecnologia", "account_age_d_conta", "iddim_contaservico_dth",
                  "account_age_d_contaservico"]]

    df = df.merge(cl_cols, left_on="codigocontaservico", right_on="codigo_contaservico", how="left")
    df = df.drop(columns=["codigo_contaservico"], errors="ignore")
    print(f"  Joined {df['iddim_cliente'].notna().sum():,} / {len(df):,} rows with client data")

    # ── Step 4: Campaign features (30-day window) ──
    print("[Step 4] Building campaign features...")
    if not df_campaigns.empty:
        camp_agg = df_campaigns.groupby("codigocontaservico").agg(
            n_contacts=("contact_ts", "count"),
            last_contact_ts=("contact_ts", "max"),
            n_successful_contacts=("sucesso_flag", "sum"),
        ).reset_index()
        camp_agg["any_success"] = (camp_agg["n_successful_contacts"] > 0).astype(int)

        df = df.merge(camp_agg, on="codigocontaservico", how="left")
        df["last_contact_ts"] = pd.to_datetime(df["last_contact_ts"], errors="coerce")
        df["days_since_last_contact"] = (ref_ts - df["last_contact_ts"].dt.tz_localize(None)).dt.days
        df["was_contacted"] = (df["n_contacts"].fillna(0) > 0).astype(int)
    else:
        print("  No campaign data found")

    for col in ["n_contacts", "n_successful_contacts", "any_success", "was_contacted"]:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)
    for col in ["last_contact_ts", "days_since_last_contact"]:
        if col not in df.columns:
            df[col] = np.nan

    print(f"  {df['was_contacted'].sum():,} rows were contacted")

    # ── Step 5: Topup features (90-day window) ──
    print("[Step 5] Building topup features...")
    if not df_topups.empty:
        # Match reference pipeline: no dedup on v1+v2 concat (keep all topups)
        df_topups["has_username"] = np.where(
            (df_topups["iddim_utilizador"] == -1) | (df_topups["iddim_utilizador"] == -100), 0, 1
        )
        df_topups["date_time"] = pd.to_datetime(df_topups["date_time"], errors="coerce").dt.tz_localize(None)

        # Join with features to get idconsumo
        feature_clients = df[["idconsumo", "iddim_cliente"]].dropna(subset=["iddim_cliente"]).drop_duplicates()
        topups_joined = feature_clients.merge(df_topups, on="iddim_cliente", how="inner")

        if not topups_joined.empty:
            agg = topups_joined.groupby("idconsumo").agg(
                topup_count=("date_time", "count"),
                topup_total_value=("valorcarregado", "sum"),
                topup_avg_value=("valorcarregado", "mean"),
                topup_std_value=("valorcarregado", "std"),
                topup_days_since_last=("date_time", "max"),
                used_selfcare=("has_username", "max"),
                topup_type_nunique=("iddim_tipo_carregamento", "nunique"),
                topup_channel_nunique=("iddim_canal", "nunique"),
            ).reset_index()

            agg["topup_cv_value"] = np.where(
                agg["topup_avg_value"] > 0,
                agg["topup_std_value"] / agg["topup_avg_value"],
                0,
            )
            agg["topup_days_since_last"] = (ref_ts - agg["topup_days_since_last"]).dt.days

            df = df.merge(agg, on="idconsumo", how="left")
            print(f"  Built topup features for {len(agg):,} contracts")
        else:
            print("  No matching topups found")
    else:
        print("  No topup data")

    topup_cols = [
        "topup_count", "topup_total_value", "topup_avg_value", "topup_std_value",
        "topup_cv_value", "topup_days_since_last", "used_selfcare",
        "topup_type_nunique", "topup_channel_nunique",
    ]
    for col in topup_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # ── Step 6: Past churns ──
    # For inference, past_churns = total churns from ALL prior expired contracts
    # for the same codigocontaservico. The current active contract hasn't expired yet,
    # so we count all churns in the history per codigocontaservico.
    print("[Step 6] Building past_churns...")
    churner_set = set(df_churners["idconsumo"].values)
    hist_sorted = hist.sort_values(["codigocontaservico", "iddim_date_fim", "idconsumo"])
    hist_sorted["is_churned"] = hist_sorted["idconsumo"].isin(churner_set).astype(int)

    # Total churns per codigocontaservico across all expired contracts
    total_churns = hist_sorted.groupby("codigocontaservico")["is_churned"].sum().reset_index()
    total_churns.columns = ["codigocontaservico", "past_churns"]

    df = df.merge(total_churns, on="codigocontaservico", how="left")
    df["past_churns"] = df["past_churns"].fillna(0).astype(int)
    print(f"  Past churns distribution: {df['past_churns'].value_counts().sort_index().head().to_dict()}")

    # ── Final cleanup ──
    # Convert date columns to match training format
    for col in ["iddim_date_inicio", "iddim_date_fim"]:
        if col in df.columns:
            df[col] = df[col].dt.date

    print(f"\nFinal shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build inference features from raw Redshift tables")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use cached raw data instead of downloading")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Number of service codes to sample")
    args = parser.parse_args()

    print("=" * 60)
    print("BUILD INFERENCE FEATURES (from raw tables)")
    print("=" * 60)

    if args.skip_download and all(os.path.exists(f) for f in CACHE_FILES.values()):
        print("\nLoading cached raw data...")
        df_parque = pd.read_parquet(CACHE_FILES["parque"])
        df_history = pd.read_parquet(CACHE_FILES["parque_history"])
        df_cliente = pd.read_parquet(CACHE_FILES["cliente"])
        df_campaigns = pd.read_parquet(CACHE_FILES["campaigns"])
        df_topups = pd.read_parquet(CACHE_FILES["topups"])
        df_churners = pd.read_parquet(CACHE_FILES["churners"])
        # Detect reference date from parque data
        ref_int = df_parque["iddim_data_parque"].max()
        reference_date = pd.to_datetime(str(ref_int), format="%Y%m%d").date()
        print(f"  Reference date: {reference_date}")
        for name, path in CACHE_FILES.items():
            df = pd.read_parquet(path)
            print(f"  {name}: {len(df):,} rows")
    else:
        df_parque, df_history, df_cliente, df_campaigns, df_topups, df_churners, reference_date = \
            download_raw_data(args.sample_size)

    # Build all features
    df_final = build_features(df_parque, df_history, df_cliente, df_campaigns, df_topups, df_churners, reference_date)

    # Save
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

    print(f"\n{'=' * 60}")
    print(f"DONE! {df_final.shape[0]:,} rows, {df_final.shape[1]} columns")
    print(f"Run inference with:")
    print(f"  python infer.py --from-registry --input {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
