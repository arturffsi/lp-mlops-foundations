#!/usr/bin/env python3
"""
Simple Redshift export script for SageMaker Processing
Mimics the working train_sagemaker_large.py pattern
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Export Redshift to S3")
    parser.add_argument("--host", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--cluster-identifier", required=True)
    parser.add_argument("--iam-role", required=True)
    parser.add_argument("--s3-export-path", required=True)
    parser.add_argument("--region", default="af-south-1")
    parser.add_argument("--table", default="dth_churn_ml_inference.inference_features")
    parser.add_argument("--sample-ratio", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows exported (for testing)")

    args = parser.parse_args()

    print("=" * 80)
    print("Redshift Export to S3")
    print("=" * 80)

    # Import redshift_connector (should be pre-installed via dependencies)
    try:
        import redshift_connector
        print("✓ redshift_connector imported successfully")
    except ImportError as e:
        print(f"ERROR: redshift_connector not found: {e}")
        print("Attempting to install...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "redshift_connector"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR installing redshift_connector: {result.stderr}")
            return 1
        import redshift_connector
        print("✓ redshift_connector installed successfully")

    # Build query
    if args.limit:
        # Use ROW_NUMBER window function to limit rows (LIMIT not supported in UNLOAD)
        select_query = f"SELECT * FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY 1) as rn FROM {args.table}) WHERE rn <= {args.limit}"
        print(f"Exporting limited sample: {args.limit:,} rows")
    elif args.sample_ratio:
        select_query = f"SELECT * FROM {args.table} WHERE RANDOM() < {args.sample_ratio}"
        print(f"Using sampling with ratio: {args.sample_ratio}")
    else:
        select_query = f"SELECT * FROM {args.table}"
        print("Exporting full table")

    # Build UNLOAD SQL
    sql = f"""
UNLOAD ('{select_query}')
TO '{args.s3_export_path}'
IAM_ROLE '{args.iam_role}'
FORMAT AS PARQUET
ALLOWOVERWRITE
PARALLEL ON
MANIFEST
REGION '{args.region}';
"""

    print(f"\n📋 Export Configuration:")
    print(f"  Host: {args.host}")
    print(f"  Database: {args.database}")
    print(f"  Table: {args.table}")
    print(f"  S3 Path: {args.s3_export_path}")
    print(f"  Region: {args.region}")
    print(f"\n📝 SQL:\n{sql}")
    print("=" * 80)

    try:
        # Connect to Redshift with IAM authentication
        print("\n🔌 Connecting to Redshift with IAM (no password)...")
        conn = redshift_connector.connect(
            iam=True,
            host=args.host,
            region=args.region,
            cluster_identifier=args.cluster_identifier,
            database=args.database,
            db_user=args.user,
            ssl=True
        )

        print("✓ Connected successfully")

        # Execute UNLOAD
        print("\n🚀 Executing UNLOAD command...")
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()

        print("✓ UNLOAD completed successfully")

        # Write success marker (only if in SageMaker Processing environment)
        output_dir = "/opt/ml/processing/output"
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/export_complete.txt", "w") as f:
                f.write(f"Export completed successfully\n")
                f.write(f"S3 Path: {args.s3_export_path}\n")
                f.write(f"Table: {args.table}\n")
            print(f"\n✓ Success marker written to {output_dir}/export_complete.txt")
        except Exception as e:
            print(f"\n⚠️  Could not write success marker (not critical): {e}")

        print("\n" + "=" * 80)
        print("✅ Redshift Export Complete")
        print("=" * 80)

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ ERROR: Redshift export failed")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
