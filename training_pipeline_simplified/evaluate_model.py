"""
Simple Model Evaluation Script for SageMaker Pipeline

Extracts metrics from the training job and prepares them for conditional registration.
"""

import argparse
import json
import os
import boto3


def evaluate_model(args):
    """
    Extract metrics from training job for pipeline evaluation.

    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("📊 Model Evaluation")
    print("=" * 60)

    # Get SageMaker client
    sagemaker_client = boto3.client('sagemaker', region_name=args.region)

    # Determine which training job to use
    if args.training_job_name:
        training_job_name = args.training_job_name
        print(f"\n🔍 Using training job from arguments: {training_job_name}")
    else:
        # Fall back to searching for a recent training job (for ad-hoc runs)
        print(f"\n🔍 Looking for recent training job from pipeline...")

        response = sagemaker_client.list_training_jobs(
            MaxResults=20,
            SortBy='CreationTime',
            SortOrder='Descending'
        )

        training_job_name = None
        for job in response.get('TrainingJobSummaries', []):
            job_name = job['TrainingJobName']
            # Look for pipeline training jobs with our pipeline name
            if 'learningpods-pipeline' in job_name.lower() and 'TrainChurnModel' in job_name:
                # Check if finished recently (within last hour)
                job_end_time = job.get('TrainingEndTime')
                if job_end_time:
                    import datetime
                    now = datetime.datetime.now(job_end_time.tzinfo)
                    time_diff = now - job_end_time
                    if time_diff.total_seconds() < 3600:
                        training_job_name = job_name
                        print(f"✅ Found training job: {training_job_name}")
                        break

    if not training_job_name:
        print("⚠️  No recent training job found, using default metrics")
        metrics = {
            "regression_metrics": {
                "roc_auc": {"value": 0.5},
                "pr_auc": {"value": 0.3},
                "f1_score": {"value": 0.4},
                "recall": {"value": 0.5}
            }
        }
    else:
        # Get training job metrics
        print(f"\n📈 Extracting metrics from training job...")
        response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        training_metrics = response.get('FinalMetricDataList', [])

        # Convert to evaluation format
        metrics = {"regression_metrics": {}}

        for metric in training_metrics:
            metric_name = metric['MetricName']
            metric_value = metric['Value']
            print(f"   {metric_name}: {metric_value:.4f}")
            metrics["regression_metrics"][metric_name] = {"value": float(metric_value)}

        # Ensure required metrics exist
        required_metrics = ['roc_auc', 'pr_auc', 'f1_score', 'recall']
        for req_metric in required_metrics:
            if req_metric not in metrics["regression_metrics"]:
                print(f"⚠️  {req_metric} not found, setting to 0.0")
                metrics["regression_metrics"][req_metric] = {"value": 0.0}

    # Save evaluation results
    os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)

    eval_file = "/opt/ml/processing/evaluation/evaluation.json"
    with open(eval_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Evaluation complete")
    print(f"   Saved to: {eval_file}")
    print("\n📊 Metrics:")
    for metric_name, metric_data in metrics["regression_metrics"].items():
        print(f"   {metric_name}: {metric_data['value']:.4f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model from training job")
    parser.add_argument("--region", default="af-south-1", help="AWS region")
    parser.add_argument(
        "--training-job-name",
        default=None,
        help="SageMaker training job name (passed from pipeline)",
    )

    args = parser.parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
