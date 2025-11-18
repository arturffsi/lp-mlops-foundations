"""
Simple Redshift Export Script for SageMaker Pipeline

Exports data from Redshift to S3 as Parquet files.
This runs as a processing step in the SageMaker Pipeline.
"""

import argparse
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv


def export_redshift_to_s3(args):
    """
    Export data from Redshift to S3.

    Args:
        args: Command line arguments with connection details
    """
    print("=" * 60)
    print("📊 Redshift Export to S3")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Get password from environment
    password = os.getenv('REDSHIFT_PASSWORD')
    if not password:
        raise RuntimeError("REDSHIFT_PASSWORD not found in environment variables")

    print(f"\n🔌 Connecting to Redshift...")
    print(f"   Host: {args.host}")
    print(f"   Database: {args.database}")
    print(f"   Table: {args.table}")

    # Connect to Redshift
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=password
    )

    try:
        # Build query with sampling
        query = f"SELECT * FROM {args.table}"
        if args.sample_ratio:
            query += f" WHERE RANDOM() < {args.sample_ratio}"

        print(f"\n📝 Query: {query}")

        # Load data
        print(f"\n⏳ Loading data from Redshift...")
        df = pd.read_sql(query, conn)
        print(f"✅ Loaded {len(df):,} rows")

    finally:
        conn.close()
        print("🔌 Connection closed")

    # Save to S3
    output_path = args.output_path
    if not output_path.endswith('/'):
        output_path += '/'

    output_file = f"{output_path}data.parquet"

    print(f"\n💾 Saving to S3: {output_file}")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"✅ Saved {len(df):,} rows to S3")

    # Write success marker for SageMaker
    try:
        os.makedirs("/opt/ml/processing/output", exist_ok=True)
        with open("/opt/ml/processing/output/export_complete.txt", "w") as f:
            f.write(f"Export completed successfully\n")
            f.write(f"Rows: {len(df):,}\n")
            f.write(f"Output: {output_file}\n")
        print("✅ Success marker written")
    except Exception as e:
        print(f"⚠️  Could not write success marker: {e}")

    print("\n" + "=" * 60)
    print("✅ Redshift Export Complete")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export Redshift data to S3")

    parser.add_argument("--host", required=True, help="Redshift host")
    parser.add_argument("--port", type=int, default=5439, help="Redshift port")
    parser.add_argument("--database", required=True, help="Redshift database")
    parser.add_argument("--user", required=True, help="Redshift user")
    parser.add_argument("--table", required=True, help="Redshift table name")
    parser.add_argument("--sample-ratio", type=float, default=None,
                       help="Sample ratio (e.g., 0.003 for 0.3%)")
    parser.add_argument("--output-path", required=True,
                       help="S3 output path (e.g., s3://bucket/path/)")

    args = parser.parse_args()

    export_redshift_to_s3(args)


if __name__ == "__main__":
    main()
