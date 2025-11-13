#!/usr/bin/env python3
"""
Upload inference results to Redshift using AWS Data Wrangler
"""

import pandas as pd
import awswrangler as wr
import boto3
from datetime import datetime
import argparse
import yaml
import os
import sys

def create_redshift_connection(config_path):
    """Create Redshift connection from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Also load inference config if it exists
    inference_config_path = config_path.replace('config.yaml', 'config.inference.yaml')
    if os.path.exists(inference_config_path):
        with open(inference_config_path, 'r') as f:
            inference_config = yaml.safe_load(f)
            config.update(inference_config)

    redshift_config = config.get('redshift_output', {})

    if 'connection_name' in redshift_config:
        # Use Glue connection
        return wr.redshift.connect(redshift_config['connection_name'])
    else:
        # Use direct connection parameters with IAM authentication
        import redshift_connector
        return redshift_connector.connect(
            iam=True,
            host=redshift_config['host'],
            region=redshift_config.get('region', 'af-south-1'),
            cluster_identifier=redshift_config['cluster_identifier'],
            database=redshift_config['database'],
            db_user=redshift_config['user'],
            ssl=True
        )

def upload_predictions_to_redshift(data_path, config_path, s3_staging_path=None):
    """
    Upload inference predictions to Redshift

    Args:
        data_path: Path to the predictions data (CSV file or directory containing CSV)
        config_path: Path to config.yaml with Redshift settings
        s3_staging_path: S3 path for staging files (optional)
    """
    print(f"📊 Starting Redshift upload process...")

    # Load the predictions
    print(f"📂 Loading predictions from: {data_path}")

    # If data_path is a directory, find the detailed_predictions CSV file
    if os.path.isdir(data_path):
        import glob
        import re
        csv_files = glob.glob(os.path.join(data_path, "detailed_predictions_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No detailed_predictions CSV files found in {data_path}")
        # Sort by timestamp in filename (format: detailed_predictions_YYYYMMDD_HHMMSS.csv)
        # Extract timestamp from filename and sort in descending order
        def get_timestamp_from_filename(filepath):
            basename = os.path.basename(filepath)
            match = re.search(r'detailed_predictions_(\d{8}_\d{6})\.csv', basename)
            return match.group(1) if match else '00000000_000000'

        csv_files.sort(key=get_timestamp_from_filename, reverse=True)
        data_path = csv_files[0]
        print(f"   Found CSV file: {os.path.basename(data_path)} (most recent of {len(csv_files)} files)")

    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} prediction records")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Also load inference config if it exists
    inference_config_path = config_path.replace('config.yaml', 'config.inference.yaml')
    if os.path.exists(inference_config_path):
        with open(inference_config_path, 'r') as f:
            inference_config = yaml.safe_load(f)
            config.update(inference_config)
        print(f"📋 Loaded inference config from: {inference_config_path}")

    redshift_config = config.get('redshift_output', {})
    if not redshift_config:
        raise ValueError("No 'redshift_output' section found in config.yaml or config.inference.yaml")

    # Set up S3 staging path
    if not s3_staging_path:
        # Use hardcoded values to avoid STS call which may not work in VPC
        account_id = '733246370304'
        region = 'af-south-1'
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        s3_staging_path = f"s3://sagemaker-{region}-{account_id}/churn-inference-staging/run_{timestamp}/"

    print(f"📁 S3 staging path: {s3_staging_path}")

    # Convert data types and rename columns
    print(f"🔄 Preparing data for Redshift...")
    df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])

    # Rename columns to match Redshift table (dots not allowed in column names)
    df = df.rename(columns={
        'churn_prediction_0.5': 'churn_prediction_0_5',
        'threshold_0.5': 'threshold_0_5'
    })

    df['churn_prediction_0_5'] = df['churn_prediction_0_5'].astype('Int64')  # Nullable integer
    df['churn_prediction_target_recall'] = df['churn_prediction_target_recall'].astype('Int64')  # Nullable integer

    # Create connection
    print(f"🔗 Connecting to Redshift...")
    con = create_redshift_connection(config_path)

    try:
        # Upload to Redshift
        schema = redshift_config.get('schema', 'public')
        table = redshift_config.get('table', 'churn_predictions')
        mode = redshift_config.get('mode', 'append')  # append or overwrite
        max_rows_per_file = redshift_config.get('max_rows_by_file', 75000)

        print(f"📤 Uploading to Redshift:")
        print(f"   Schema: {schema}")
        print(f"   Table: {table}")
        print(f"   Mode: {mode}")
        print(f"   Rows per file: {max_rows_per_file:,}")

        # Check if table exists, create if it doesn't
        cursor = con.cursor()
        try:
            # Handle schema.table format
            table_parts = table.split('.')
            if len(table_parts) == 2:
                check_schema, check_table = table_parts
            else:
                check_schema, check_table = schema, table

            cursor.execute(f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = '{check_schema}'
                AND table_name = '{check_table}'
            """)
            table_exists = cursor.fetchone()[0] > 0

            if not table_exists:
                print(f"📋 Table {schema}.{table} does not exist, creating...")
                create_table_sql = f"""
                CREATE TABLE {schema}.{table} (
                    idconsumo BIGINT,
                    codigocontaservico BIGINT,
                    churn_probability FLOAT,
                    churn_prediction_0_5 INTEGER,
                    churn_prediction_target_recall INTEGER,
                    prediction_timestamp TIMESTAMP,
                    model_version VARCHAR(50),
                    threshold_0_5 DECIMAL(3,1),
                    threshold_target_recall FLOAT
                );
                """
                cursor.execute(create_table_sql)
                con.commit()
                print(f"✅ Table {schema}.{table} created successfully")
        finally:
            cursor.close()

        # Prepare dtype mapping for Redshift
        dtype_mapping = {
            'idconsumo': 'BIGINT',
            'codigocontaservico': 'BIGINT',
            'churn_probability': 'FLOAT',
            'churn_prediction_0_5': 'INTEGER',
            'churn_prediction_target_recall': 'INTEGER',
            'prediction_timestamp': 'TIMESTAMP',
            'model_version': 'VARCHAR(50)',
            'threshold_0_5': 'DECIMAL(3,1)',
            'threshold_target_recall': 'FLOAT'
        }

        # Use awswrangler to copy data
        result = wr.redshift.copy(
            df=df,
            path=s3_staging_path,
            con=con,
            schema=schema,
            table=table,
            mode=mode,
            dtype=dtype_mapping,
            max_rows_by_file=max_rows_per_file,
            use_threads=True
        )

        print(f"✅ Successfully uploaded {len(df):,} records to Redshift!")
        print(f"   Table: {schema}.{table}")

        return result

    except Exception as e:
        print(f"❌ Error uploading to Redshift: {str(e)}")
        raise
    finally:
        if hasattr(con, 'close'):
            con.close()

def main():
    parser = argparse.ArgumentParser(description="Upload inference results to Redshift")
    parser.add_argument("--data-path", required=True,
                       help="Path to the predictions data (directory of parquet files)")
    parser.add_argument("--config", default="../config.yaml",
                       help="Path to config.yaml file")
    parser.add_argument("--s3-staging-path",
                       help="S3 path for staging files (auto-generated if not provided)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode - validate connection and data only")

    args = parser.parse_args()

    if args.test:
        print("🧪 Test mode - validating configuration and data...")

        # Check if data path exists
        if not os.path.exists(args.data_path):
            print(f"❌ Data path not found: {args.data_path}")
            return

        # Check if config exists
        if not os.path.exists(args.config):
            print(f"❌ Config file not found: {args.config}")
            return

        # Load and validate config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        redshift_config = config.get('redshift_output', {})
        if not redshift_config:
            print(f"❌ No 'redshift_output' section in config")
            return

        # Load and validate Parquet data
        try:
            df = pd.read_parquet(args.data_path)
            print(f"✅ Parquet validation passed: {len(df):,} rows")
            print(f"   Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Parquet validation failed: {e}")
            return

        print(f"✅ Configuration validation passed")
        print(f"   Schema: {redshift_config.get('schema', 'public')}")
        print(f"   Table: {redshift_config.get('table', 'churn_predictions')}")

    else:
        # Run the actual upload
        upload_predictions_to_redshift(
            data_path=args.data_path,
            config_path=args.config,
            s3_staging_path=args.s3_staging_path
        )

if __name__ == "__main__":
    main()
