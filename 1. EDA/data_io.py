#!/usr/bin/env python3
"""
Data loading utilities for EDA - supports both local/S3 parquet and Redshift

Usage:
    # Load from local parquet
    df = load_data(source='parquet', uri='../sample_data_from_redshift/sample.parquet')

    # Load from S3 parquet
    df = load_data(source='parquet', uri='s3://bucket/path/data.parquet')

    # Load from Redshift (boto3 redshift-data API + UNLOAD to S3)
    df = load_data(
        source='redshift',
        sql='SELECT * FROM dth_churn_ml_inference.inference_features',
        redshift_kwargs={
            'cluster_id': 'redshift-cluster-dsi',
            'database': 'prod',
            'db_user': 'svc_sagemaker',
            'region': 'af-south-1',
            'iam_role': 'arn:aws:iam::733246370304:role/RedshiftIAMAuthRole',
            's3_export_prefix': 's3://bucket/path/export/'
        },
        limit=100000
    )
"""
import os
import pandas as pd
from typing import Optional, Dict


def load_data(
    source: str,
    uri: Optional[str] = None,
    sql: Optional[str] = None,
    redshift_kwargs: Optional[Dict] = None,
    limit: Optional[int] = None,
    sample_ratio: Optional[float] = None
) -> pd.DataFrame:
    """
    Load data from parquet or Redshift

    Parameters:
    -----------
    source : str
        Data source type: 'parquet' or 'redshift'
    uri : str, optional
        For parquet: local path or s3:// path
        For redshift: ignored (connection info in redshift_kwargs)
    sql : str, optional
        SQL query for Redshift (required when source='redshift')
    redshift_kwargs : dict, optional
        Connection parameters for Redshift (boto3 redshift-data API):
        - cluster_id: Redshift cluster identifier (e.g., 'redshift-cluster-dsi')
        - database: database name (e.g., 'prod')
        - db_user: database user for IAM auth (e.g., 'svc_sagemaker')
        - region: AWS region (e.g., 'af-south-1')
        - iam_role: ARN of IAM role for S3 access (required for UNLOAD)
        - s3_export_prefix: S3 path prefix for UNLOAD (e.g., 's3://bucket/path/')
        - cleanup_s3: Whether to delete S3 files after loading (default: False)
    limit : int, optional
        Limit number of rows (applies to both parquet and redshift)
    sample_ratio : float, optional
        Sample ratio 0-1 (only applies to parquet)

    Returns:
    --------
    pd.DataFrame
        Loaded data
    """

    if source == "parquet":
        return _load_from_parquet(uri, limit, sample_ratio)
    elif source == "redshift":
        return _load_from_redshift(sql, redshift_kwargs, limit)
    else:
        raise ValueError(f"Unknown source: {source}. Must be 'parquet' or 'redshift'")


def _load_from_parquet(
    uri: str,
    limit: Optional[int] = None,
    sample_ratio: Optional[float] = None
) -> pd.DataFrame:
    """Load data from parquet file (local or S3)"""

    if not uri:
        raise ValueError("URI must be provided for parquet source")

    print(f"📂 Loading from parquet: {uri}")

    # Check if S3 path
    if uri.startswith("s3://"):
        try:
            import s3fs
        except ImportError:
            raise ImportError(
                "s3fs is required for S3 access. Install with: pip install s3fs"
            )

        # Handle wildcards for multiple files
        if "*" in uri:
            fs = s3fs.S3FileSystem()
            paths = fs.glob(uri)
            print(f"  Found {len(paths)} files matching pattern")
            dfs = [pd.read_parquet(f"s3://{p}", engine="pyarrow") for p in paths]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(uri, engine="pyarrow")
    else:
        # Local file
        df = pd.read_parquet(uri, engine="pyarrow")

    print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Apply sampling if requested
    if sample_ratio is not None:
        if not 0 < sample_ratio <= 1:
            raise ValueError("sample_ratio must be between 0 and 1")
        original_size = len(df)
        df = df.sample(frac=sample_ratio, random_state=42)
        print(f"  ✓ Sampled {sample_ratio:.1%}: {len(df):,} rows (from {original_size:,})")

    # Apply limit if requested
    if limit is not None:
        if limit < len(df):
            df = df.head(limit)
            print(f"  ✓ Limited to {limit:,} rows")

    return df


def _load_from_redshift(
    sql: str,
    redshift_kwargs: Dict,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load data from Redshift using UNLOAD to S3 (matches test_redshift_connection.ipynb pattern)

    This follows the EXACT pattern from test_redshift_connection.ipynb:
    1. Use boto3 redshift-data API (no direct connection needed)
    2. Execute UNLOAD command to export data to S3 as Parquet
    3. Poll for query completion
    4. Read Parquet files from S3 using awswrangler
    """

    if not sql:
        raise ValueError("SQL query must be provided for redshift source")

    if not redshift_kwargs:
        raise ValueError("redshift_kwargs must be provided for redshift source")

    # Validate required parameters (matching test notebook)
    required_params = ['cluster_id', 'database', 'db_user', 'region', 'iam_role', 's3_export_prefix']
    missing = [p for p in required_params if p not in redshift_kwargs]
    if missing:
        raise ValueError(f"Missing required redshift_kwargs: {', '.join(missing)}")

    # Import dependencies
    try:
        import boto3
        import time
    except ImportError:
        raise ImportError(
            "boto3 is required for Redshift access. "
            "Install with: pip install boto3"
        )

    try:
        import awswrangler as wr
    except ImportError:
        raise ImportError(
            "awswrangler is required for S3 access. Install with: pip install awswrangler"
        )

    print("=" * 80)
    print("📊 Redshift UNLOAD Export (boto3 redshift-data API)")
    print("=" * 80)
    print(f"  Cluster ID: {redshift_kwargs['cluster_id']}")
    print(f"  Database: {redshift_kwargs['database']}")
    print(f"  User: {redshift_kwargs['db_user']}")
    print(f"  Region: {redshift_kwargs['region']}")
    print(f"  S3 Export Prefix: {redshift_kwargs['s3_export_prefix']}")

    # Build UNLOAD SQL (exactly like test notebook)
    region = redshift_kwargs['region']
    s3_export_prefix = redshift_kwargs['s3_export_prefix'].rstrip('/')
    iam_role = redshift_kwargs['iam_role']

    # Clean up SQL query (remove trailing semicolons and whitespace)
    inner_sql = sql.rstrip().rstrip(';').rstrip()

    # NOTE: Redshift UNLOAD does NOT support LIMIT clause inside the query
    # We'll apply the limit after loading from S3 instead
    if limit is not None:
        print(f"  ℹ️  Will limit to {limit:,} rows after loading (UNLOAD doesn't support LIMIT)")

    unload_sql = f"""
UNLOAD ('
  {inner_sql}
')
TO '{s3_export_prefix}/'
IAM_ROLE '{iam_role}'
FORMAT AS PARQUET
ALLOWOVERWRITE
PARALLEL ON
REGION '{region}';
"""

    print(f"\n📝 UNLOAD SQL:")
    print(unload_sql)
    print("=" * 80)

    try:
        # Execute query using boto3 redshift-data API (same as test notebook)
        print("\n🚀 Executing UNLOAD via redshift-data API...")
        client = boto3.client("redshift-data", region_name=region)

        resp = client.execute_statement(
            ClusterIdentifier=redshift_kwargs['cluster_id'],
            Database=redshift_kwargs['database'],
            DbUser=redshift_kwargs['db_user'],
            Sql=unload_sql
        )

        stmt_id = resp["Id"]
        print(f"  Statement ID: {stmt_id}")

        # Poll for completion (same as test notebook)
        print("\n⏳ Waiting for UNLOAD to complete...")
        while True:
            desc = client.describe_statement(Id=stmt_id)
            status = desc["Status"]
            print(f"  Status: {status}")

            if status in ["FINISHED", "FAILED", "ABORTED"]:
                break
            time.sleep(2)

        if status != "FINISHED":
            error_msg = desc.get("Error", "Unknown error")
            raise RuntimeError(f"UNLOAD failed with status {status}: {error_msg}")

        print("  ✓ UNLOAD completed successfully")

        # Read parquet files from S3 using awswrangler (same as test notebook)
        print(f"\n📂 Reading Parquet files from S3: {s3_export_prefix}/")

        files = wr.s3.list_objects(s3_export_prefix)
        print(f"  Found {len(files)} objects")

        if len(files) == 0:
            raise RuntimeError(f"No files found at {s3_export_prefix}/")

        # Show first few files
        for path in files[:5]:
            print(f"    {path}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")

        # Read all parquet files as dataset
        df = wr.s3.read_parquet(s3_export_prefix, dataset=True)

        print(f"\n  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns from S3")

        # Apply limit if requested (after loading, since UNLOAD doesn't support LIMIT)
        if limit is not None and len(df) > limit:
            df = df.head(limit)
            print(f"  ✓ Limited to {limit:,} rows")

        # Cleanup S3 files if requested
        if redshift_kwargs.get('cleanup_s3', False):
            print(f"\n🧹 Cleaning up S3 files...")
            wr.s3.delete_objects(s3_export_prefix)
            print("  ✓ Cleanup complete")

        print("\n" + "=" * 80)
        print("✅ Redshift UNLOAD Export Complete")
        print("=" * 80)

        return df

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR: Redshift UNLOAD export failed")
        print("=" * 80)
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def save_metadata(df: pd.DataFrame, source: str, output_path: str = "data_metadata.json"):
    """
    Save data provenance metadata for MLOps tracking

    Parameters:
    -----------
    df : pd.DataFrame
        The loaded dataframe
    source : str
        Source type ('parquet' or 'redshift')
    output_path : str
        Where to save the metadata JSON
    """
    import json
    from datetime import datetime, UTC

    metadata = {
        "source": source,
        "extraction_date": datetime.now(UTC).isoformat(),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "version": "v1.0"
    }

    # Add date range if date columns exist
    date_cols = df.select_dtypes(include='datetime').columns
    if len(date_cols) > 0:
        first_date_col = date_cols[0]
        metadata["date_range"] = {
            "column": first_date_col,
            "min": str(df[first_date_col].min()),
            "max": str(df[first_date_col].max())
        }

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to {output_path}")

    return metadata


if __name__ == "__main__":
    # Example usage
    print("Example 1: Load from local parquet")
    print("-" * 60)
    print("df = load_data(")
    print("    source='parquet',")
    print("    uri='../sample_data_from_redshift/sample.parquet',")
    print("    limit=1000")
    print(")")

    print("\n" + "=" * 60)
    print("Example 2: Load from Redshift with UNLOAD (boto3 redshift-data API)")
    print("-" * 60)
    print("df = load_data(")
    print("    source='redshift',")
    print("    sql='SELECT * FROM dth_churn_ml_training.training_features',")
    print("    redshift_kwargs={")
    print("        'cluster_id': 'redshift-cluster-dsi',")
    print("        'database': 'prod',")
    print("        'db_user': 'svc_sagemaker',")
    print("        'region': 'af-south-1',")
    print("        'iam_role': 'arn:aws:iam::733246370304:role/RedshiftIAMAuthRole',")
    print("        's3_export_prefix': 's3://bucket/path/export/',")
    print("        'cleanup_s3': False  # Set to True to delete temp files")
    print("    },")
    print("    limit=100000")
    print(")")
