#!/usr/bin/env python3
"""
Register a model in SageMaker Model Registry from a completed training job.
This creates versioned models with rich metadata, governance, and approval workflows.
"""

import argparse
import boto3
from datetime import datetime
import sys
import os

# Add utils directory to path to import job_management
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from job_management import register_model_in_registry, list_recent_training_jobs, list_model_packages














def main():
    parser = argparse.ArgumentParser(description="Register model in SageMaker Model Registry")
    parser.add_argument("--training-job-name", required=False, 
                       help="Name of the completed training job")
    parser.add_argument("--model-package-group", default="churn-model-group",
                       help="Model package group name for versioning")
    parser.add_argument("--approval-status", 
                       choices=['Approved', 'Rejected', 'PendingManualApproval'],
                       default='PendingManualApproval',
                       help="Initial approval status")
    parser.add_argument("--list-models", action="store_true",
                       help="List models in registry")
    parser.add_argument("--list-jobs", action="store_true",
                       help="List recent training jobs")
    
    args = parser.parse_args()
    
    print("🏛️  SageMaker Model Registry")
    print("=" * 60)
    
    # List recent training jobs if needed
    if args.list_jobs or not args.training_job_name:
        try:
            list_recent_training_jobs()
        except Exception as e:
            print(f"📋 Recent training jobs listing not available: {e}")
        
        if not args.training_job_name and not args.list_models:
            print("\n💡 Choose a training job name and run:")
            print("   python src/create_model_registry_from_job.py --training-job-name [JOB_NAME]")
            return
    
    if args.list_models:
        list_model_packages(args.model_package_group)
    
    if args.training_job_name:
        # Register model in registry
        model_package_arn = register_model_in_registry(
            training_job_name=args.training_job_name,
            model_package_group_name=args.model_package_group,
            approval_status=args.approval_status
        )
        
        if model_package_arn:
            print(f"\n🎉 Model registration completed successfully!")
            print(f"🔗 Next: View and approve in Model Registry console")
        else:
            print(f"\n💔 Model registration failed. Check the error messages above.")


if __name__ == "__main__":
    main()