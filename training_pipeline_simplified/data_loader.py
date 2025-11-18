"""
Simplified Data Loader for SageMaker
Handles loading data from Parquet or Redshift with clean, beginner-friendly code.
Follows the pattern from load_redshift.ipynb
"""

import os
import pandas as pd
from dotenv import load_dotenv


def load_from_redshift(sql, redshift_kwargs):
    """
    Load data from Redshift database using psycopg2.
    Follows the pattern from load_redshift.ipynb

    Args:
        sql: SQL query to execute
        redshift_kwargs: Connection parameters (host, port, dbname, user, password)

    Returns:
        DataFrame with data from Redshift
    """
    print(f"📊 Loading data from Redshift...")
    print(f"   Host: {redshift_kwargs['host']}")
    print(f"   Database: {redshift_kwargs['dbname']}")
    print(f"   SQL: {sql[:100]}...")  # Print first 100 chars of SQL

    import psycopg2

    # Load environment variables (for REDSHIFT_PASSWORD)
    load_dotenv()

    # Get password from environment variable if using ${...} syntax
    password = redshift_kwargs['password']
    if password.startswith('${') and password.endswith('}'):
        env_var = password[2:-1]  # Extract variable name
        password = os.getenv(env_var)
        if not password:
            raise RuntimeError(f"{env_var} not found in environment variables. "
                             f"Please create a .env file with {env_var}=your_password")

    conn = psycopg2.connect(
        host=redshift_kwargs['host'],
        port=redshift_kwargs['port'],
        dbname=redshift_kwargs['dbname'],
        user=redshift_kwargs['user'],
        password=password
    )

    try:
        df = pd.read_sql(sql, conn)
        print(f"✅ Loaded {len(df):,} rows from Redshift")
        return df
    finally:
        conn.close()


def load_from_parquet(uri):
    """
    Load data from Parquet file(s).
    Handles both local files and S3 URIs with multiple files.
    Works in SageMaker Pipeline (data pre-downloaded) or standalone.

    Args:
        uri: Path to parquet file, directory, or S3 URI

    Returns:
        DataFrame with data from Parquet
    """
    print(f"📊 Loading data from Parquet: {uri}")

    # Check if running in SageMaker Pipeline (data is already downloaded)
    sagemaker_data_dir = "/opt/ml/input/data/training"
    if os.path.exists(sagemaker_data_dir):
        print(f"   Running in SageMaker Pipeline - using mounted data directory")
        import glob
        parquet_files = glob.glob(f"{sagemaker_data_dir}/*.parquet")

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {sagemaker_data_dir}")

        print(f"   Found {len(parquet_files)} parquet file(s)")

        # Load and concatenate all parquet files
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

    else:
        # Running locally - load directly from S3 or local path
        print(f"   Running locally - loading from: {uri}")

        if uri.startswith('s3://'):
            # For S3 URIs, use boto3 to list and load all parquet files
            import boto3

            # Parse S3 URI
            uri_parts = uri.replace('s3://', '').split('/', 1)
            bucket = uri_parts[0]
            prefix = uri_parts[1] if len(uri_parts) > 1 else ''

            # Ensure prefix doesn't start with / but ends with / for directory listing
            prefix = prefix.lstrip('/').rstrip('/') + '/' if prefix else ''

            print(f"   S3 Bucket: {bucket}")
            print(f"   S3 Prefix: {prefix}")

            # Create boto3 session with specific profile
            # Try AdministratorAccess first, fall back to DataScientist or default
            try:
                session = boto3.Session(profile_name="AdministratorAccess-733246370304")
                print(f"   Using AWS profile: AdministratorAccess-733246370304")
            except:
                try:
                    session = boto3.Session(profile_name="DataScientist-733246370304")
                    print(f"   Using AWS profile: DataScientist-733246370304")
                except:
                    # Fall back to default credentials (for SageMaker environment)
                    session = boto3.Session()
                    print(f"   Using default AWS credentials")

            # List all parquet files in the S3 location
            s3 = session.client('s3')
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if 'Contents' not in response:
                raise FileNotFoundError(f"No files found at {uri}")

            # Filter for parquet files
            parquet_keys = [obj['Key'] for obj in response['Contents']
                          if obj['Key'].endswith('.parquet')]

            if not parquet_keys:
                raise FileNotFoundError(f"No parquet files found at {uri}")

            print(f"   Found {len(parquet_keys)} parquet file(s)")

            # Load and concatenate all parquet files
            # Use s3fs with the boto3 session for authentication
            import s3fs
            fs = s3fs.S3FileSystem(session=session)

            dfs = []
            for key in parquet_keys:
                s3_path = f"s3://{bucket}/{key}"
                print(f"   Loading: {key}")
                # Read using s3fs filesystem with proper credentials
                with fs.open(s3_path, 'rb') as f:
                    dfs.append(pd.read_parquet(f))

            df = pd.concat(dfs, ignore_index=True)
        else:
            # Local file or directory
            df = pd.read_parquet(uri)

    print(f"✅ Loaded {len(df):,} rows from Parquet")
    return df


def load_data(config):
    """
    Load data based on configuration.

    In SageMaker Pipeline mode: Loads from S3 (exported by Redshift export step)
    In standalone mode: Can load from S3, Parquet, or Redshift

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        DataFrame with loaded data
    """
    print(f"\n{'='*60}")
    print(f"🔄 Loading data")
    print(f"{'='*60}")

    # Check if running in SageMaker Pipeline (data pre-downloaded to /opt/ml/input/data/training)
    sagemaker_data_dir = "/opt/ml/input/data/training"
    if os.path.exists(sagemaker_data_dir):
        print(f"📦 SageMaker Pipeline mode: Loading from pre-downloaded data")
        # In pipeline mode, data was exported from Redshift to S3, then downloaded by SageMaker
        df = load_from_parquet(config['data'].get('s3_uri', config['data']['parquet_uri']))
    else:
        # Standalone mode - use config to determine source
        source = config['data']['source']
        print(f"🖥️  Standalone mode: Loading from {source}")

        if source == 's3':
            # Load directly from S3 (fastest, no VPN needed)
            df = load_from_parquet(config['data']['s3_uri'])
        elif source == 'parquet':
            df = load_from_parquet(config['data']['parquet_uri'])
        elif source == 'redshift':
            df = load_from_redshift(
                config['data']['redshift_sql'],
                config['data']['redshift_kwargs']
            )
        else:
            raise ValueError(f"Unknown data source: {source}. Use 's3', 'parquet', or 'redshift'")

    print(f"📊 Data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*60}\n")

    return df
