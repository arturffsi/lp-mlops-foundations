#!/usr/bin/env python3
"""
Data loading utilities for EDA - supports both local/S3 parquet and Redshift

Usage:
    # Load from local parquet
    df = load_data(source='parquet', uri='../sample_data_from_redshift/sample.parquet')

    # Load from S3 parquet
    df = load_data(source='parquet', uri='s3://bucket/path/data.parquet')

    # Load from Redshift (IAM auth + UNLOAD to S3)
    df = load_data(
        source='redshift',
        sql='SELECT * FROM dth_churn_ml_inference.inference_features',
        redshift_kwargs={
            'host': 'your-cluster.region.redshift.amazonaws.com',
            'database': 'your_database',
            'user': 'your_user',
            'cluster_identifier': 'your-cluster',
            'region': 'af-south-1',
            'iam': True,  # Use IAM authentication
            'iam_role': 'arn:aws:iam::account:role/RedshiftS3Role',  # For S3 access
            's3_export_path': 's3://your-bucket/tmp/eda_export/'  # Temp S3 location
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
        Connection parameters for Redshift:
        - host: Redshift endpoint
        - database: database name
        - user: db user (for IAM auth)
        - cluster_identifier: cluster ID (for IAM auth)
        - region: AWS region (default: 'af-south-1')
        - iam: True for IAM auth, False for password auth
        - iam_role: ARN of IAM role for S3 access (required for redshift)
        - s3_export_path: S3 path for UNLOAD (required for redshift)
        - cleanup_s3: Whether to delete S3 files after loading (default: False)
        - password: required if iam=False
        - port: default 5439
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
    Load data from Redshift using UNLOAD to S3 (matches production pattern)

    This mimics the production export process:
    1. Connect to Redshift with IAM auth
    2. Execute UNLOAD command to export data to S3 as Parquet
    3. Read Parquet files from S3
    4. Optionally clean up temp files
    """

    if not sql:
        raise ValueError("SQL query must be provided for redshift source")

    if not redshift_kwargs:
        raise ValueError("redshift_kwargs must be provided for redshift source")

    # Validate required parameters
    required_params = ['host', 'database', 'user', 'cluster_identifier', 'iam_role', 's3_export_path']
    missing = [p for p in required_params if p not in redshift_kwargs]
    if missing:
        raise ValueError(f"Missing required redshift_kwargs: {', '.join(missing)}")

    # Import dependencies
    try:
        import redshift_connector
    except ImportError:
        raise ImportError(
            "redshift_connector is required for Redshift access. "
            "Install with: pip install redshift-connector"
        )

    try:
        import s3fs
    except ImportError:
        raise ImportError(
            "s3fs is required for S3 access. Install with: pip install s3fs"
        )

    # Apply limit to SQL if requested (use ROW_NUMBER for UNLOAD compatibility)
    if limit is not None:
        # UNLOAD doesn't support LIMIT, so use ROW_NUMBER window function
        sql = f"SELECT * FROM ({sql.rstrip(';')}) AS subquery WHERE ROW_NUMBER() OVER (ORDER BY 1) <= {limit}"
        print(f"  ℹ️  Limited query to {limit:,} rows using ROW_NUMBER()")

    print("=" * 80)
    print("📊 Redshift UNLOAD Export (Production Pattern)")
    print("=" * 80)
    print(f"  Host: {redshift_kwargs['host']}")
    print(f"  Database: {redshift_kwargs['database']}")
    print(f"  User: {redshift_kwargs['user']}")
    print(f"  Auth: {'IAM' if redshift_kwargs.get('iam', True) else 'Password'}")
    print(f"  S3 Export Path: {redshift_kwargs['s3_export_path']}")

    # Build UNLOAD SQL (exactly like production)
    region = redshift_kwargs.get('region', 'af-south-1')
    s3_export_path = redshift_kwargs['s3_export_path'].rstrip('/')
    iam_role = redshift_kwargs['iam_role']

    unload_sql = f"""
UNLOAD ('{sql}')
TO '{s3_export_path}/'
IAM_ROLE '{iam_role}'
FORMAT AS PARQUET
ALLOWOVERWRITE
PARALLEL ON
MANIFEST
REGION '{region}';
"""

    print(f"\n📝 UNLOAD SQL:")
    print(unload_sql)
    print("=" * 80)

    # Prepare connection parameters (IAM auth only for production pattern)
    conn_params = {
        'host': redshift_kwargs['host'],
        'database': redshift_kwargs['database'],
        'ssl': True,
        'iam': True,
        'db_user': redshift_kwargs['user'],
        'cluster_identifier': redshift_kwargs['cluster_identifier'],
        'region': region
    }

    try:
        # Connect to Redshift
        print("\n🔌 Connecting to Redshift with IAM...")
        conn = redshift_connector.connect(**conn_params)
        print("  ✓ Connected successfully")

        # Execute UNLOAD
        print("\n🚀 Executing UNLOAD command...")
        cursor = conn.cursor()
        cursor.execute(unload_sql)
        conn.commit()
        cursor.close()
        conn.close()
        print("  ✓ UNLOAD completed successfully")

        # Read parquet files from S3
        print(f"\n📂 Reading Parquet files from S3: {s3_export_path}/")
        fs = s3fs.S3FileSystem()

        # Remove s3:// prefix for s3fs
        s3_path_clean = s3_export_path.replace('s3://', '')

        # Find all parquet files
        parquet_files = fs.glob(f"{s3_path_clean}/*.parquet")

        if not parquet_files:
            raise RuntimeError(f"No parquet files found at {s3_export_path}/")

        print(f"  Found {len(parquet_files)} parquet file(s)")

        # Read all parquet files
        dfs = []
        for i, path in enumerate(parquet_files, 1):
            print(f"  Reading file {i}/{len(parquet_files)}: {path.split('/')[-1]}")
            dfs.append(pd.read_parquet(f"s3://{path}", engine="pyarrow"))

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)
        print(f"\n  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

        # Cleanup S3 files if requested
        if redshift_kwargs.get('cleanup_s3', False):
            print(f"\n🧹 Cleaning up S3 files...")
            for path in parquet_files:
                fs.rm(f"s3://{path}")
            # Also remove manifest file if it exists
            manifest_file = f"{s3_path_clean}/manifest"
            if fs.exists(manifest_file):
                fs.rm(manifest_file)
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
    print("Example 2: Load from Redshift with UNLOAD (Production Pattern)")
    print("-" * 60)
    print("df = load_data(")
    print("    source='redshift',")
    print("    sql='SELECT * FROM dth_churn_ml_inference.inference_features',")
    print("    redshift_kwargs={")
    print("        'host': 'your-cluster.af-south-1.redshift.amazonaws.com',")
    print("        'database': 'your_database',")
    print("        'user': 'your_user',")
    print("        'cluster_identifier': 'your-cluster',")
    print("        'region': 'af-south-1',")
    print("        'iam': True,")
    print("        'iam_role': 'arn:aws:iam::123456789012:role/RedshiftS3Role',")
    print("        's3_export_path': 's3://your-bucket/tmp/eda_export',")
    print("        'cleanup_s3': False  # Set to True to delete temp files")
    print("    },")
    print("    limit=100000")
    print(")")
