"""
Simple Redshift Export Script for SageMaker Pipeline

Exports data from Redshift to S3 as Parquet files using UNLOAD.
This runs as a processing step in the SageMaker Pipeline.
"""

import argparse
import os
import sys
import traceback


def export_redshift_to_s3(args):
    """
    Export data from Redshift to S3 using UNLOAD command.

    Args:
        args: Command line arguments with connection details
    """
    print("=" * 80)
    print("📊 Redshift Export to S3")
    print("=" * 80)

    # Import redshift_connector
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

    # Build query with sampling
    # Use modulo sampling on idconsumo (much faster than RANDOM())
    if args.modulo_divisor:
        select_query = f"SELECT * FROM {args.table} WHERE idconsumo % {args.modulo_divisor} = 0"
        print(f"Using modulo sampling: idconsumo % {args.modulo_divisor} = 0 (~{100/args.modulo_divisor:.2f}% sample)")
    elif args.sample_ratio:
        select_query = f"SELECT * FROM {args.table} WHERE RANDOM() < {args.sample_ratio}"
        print(f"Using RANDOM() sampling with ratio: {args.sample_ratio}")
    else:
        select_query = f"SELECT * FROM {args.table}"
        print("Exporting full table")

    # Build UNLOAD SQL
    sql = f"""
UNLOAD ('{select_query}')
TO '{args.output_path}'
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
    print(f"  S3 Path: {args.output_path}")
    print(f"  Region: {args.region}")
    print(f"\n📝 SQL:\n{sql}")
    print("=" * 80)

    try:
        # Connect to Redshift with IAM authentication (no password needed!)
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

        # Write success marker
        output_dir = "/opt/ml/processing/output"
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/export_complete.txt", "w") as f:
                f.write(f"Export completed successfully\n")
                f.write(f"S3 Path: {args.output_path}\n")
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
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description="Export Redshift data to S3")

    parser.add_argument("--host", required=True, help="Redshift host")
    parser.add_argument("--database", required=True, help="Redshift database")
    parser.add_argument("--user", required=True, help="Redshift user")
    parser.add_argument("--cluster-identifier", required=True, help="Redshift cluster identifier")
    parser.add_argument("--iam-role", required=True, help="IAM role for UNLOAD")
    parser.add_argument("--table", required=True, help="Redshift table name")
    parser.add_argument("--modulo-divisor", type=int, default=None,
                       help="Modulo divisor for sampling (e.g., 333 for ~0.3%% sample)")
    parser.add_argument("--sample-ratio", type=float, default=None,
                       help="Sample ratio using RANDOM() - slower, use modulo-divisor instead")
    parser.add_argument("--output-path", required=True,
                       help="S3 output path (e.g., s3://bucket/path/)")
    parser.add_argument("--region", default="af-south-1", help="AWS region")

    args = parser.parse_args()

    try:
        exit_code = export_redshift_to_s3(args)
    except Exception as e:
        print(f"\nUNCAUGHT EXCEPTION: {e}")
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
