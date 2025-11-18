#!/usr/bin/env python3
"""
Simple SageMaker Pipeline following the working scheduled_training_large.py pattern
Just Training -> Evaluation -> Conditional Registration (no separate preprocessing)
"""

import os
from datetime import datetime
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics
import yaml


def create_simple_churn_pipeline(
    region="af-south-1",
    pipeline_name="churn-training-pipeline-v2",
    model_package_group_name="churn-model-package-group-v2"
):
    """
    Create Simple SageMaker Pipeline following scheduled_training_large.py pattern
    """
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = "sagemaker-af-south-1-733246370304"
    pipeline_prefix = f"s3://{bucket}/{pipeline_name}"

    # Load config to get data source (same as working script)
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_source = config['data']['source']

    # Create datestamp for this run (YYYY-MM-DD format)
    datestamp = datetime.now().strftime("%Y-%m-%d")

    # Redshift configuration
    redshift_host = "zap-airflow-endpoint-omdjytkjkmzjhjkmiaoe.cl4o4mmtx9ir.af-south-1.redshift.amazonaws.com"
    redshift_database = "prod"
    redshift_user = "svc_sagemaker"
    redshift_cluster_id = "redshift-cluster-dsi"
    redshift_iam_role = "arn:aws:iam::733246370304:role/RedshiftIAMAuthRole"
    redshift_export_path = f"s3://{bucket}/redshift_exports/prod_train_{datestamp}/"

    print(f"Data source: {data_source}")
    print(f"Datestamp: {datestamp}")
    print(f"Redshift export path: {redshift_export_path}")

    # Parameters
    # For 50M rows production dataset - tested with 7M rows using 1.6GB peak memory
    # 50M / 7M = ~7x data, estimate ~12GB needed + overhead = ml.m5.4xlarge (64GB) for safety
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.4xlarge"  # 64GB RAM for 50M row production dataset
    )

    roc_auc_threshold = ParameterFloat(
        name="RocAucThreshold",
        default_value=0.7
    )

    pr_auc_threshold = ParameterFloat(
        name="PrAucThreshold",
        default_value=0.5
    )

    f1_threshold = ParameterFloat(
        name="F1Threshold",
        default_value=0.55
    )

    churner_recall_threshold = ParameterFloat(
        name="ChurnerRecallThreshold",
        default_value=0.7
    )

    # =============================================================================
    # Step 0: Redshift Export to S3
    # =============================================================================

    # Create Redshift export processor - use ScriptProcessor with PyTorch image
    from sagemaker.processing import ScriptProcessor, ProcessingInput as ProcInput
    from sagemaker import image_uris

    # Get PyTorch processing image
    pytorch_image = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.0.0",
        py_version="py310",
        instance_type="ml.m5.large",
        image_scope="training"
    )

    # Use ScriptProcessor with bash bootstrap script to install dependencies
    # Add VPC configuration for Redshift access
    from sagemaker.network import NetworkConfig

    network_config = NetworkConfig(
        enable_network_isolation=False,  # Allow S3 access via VPC endpoint
        subnets=["subnet-06cf22533465bffc6", "subnet-04a897538d1ac1390"],  # Private subnets in zap VPC
        security_group_ids=["sg-017fb44c1ff043deb"],  # sm-vpce-sg (has Redshift port 5439 access)
    )

    redshift_processor = ScriptProcessor(
        image_uri=pytorch_image,
        command=["bash"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{pipeline_name}-redshift-export",
        network_config=network_config,
    )

    # Paths
    bootstrap_script_path = os.path.join(os.path.dirname(__file__), "..", "redshift_export", "bootstrap.sh")
    export_script_path = os.path.join(os.path.dirname(__file__), "..", "redshift_export", "export_to_s3.py")

    # Redshift export processing step with bootstrap
    from sagemaker.workflow.steps import ProcessingStep

    step_redshift_export = ProcessingStep(
        name="RedshiftExport",
        processor=redshift_processor,
        inputs=[
            ProcInput(
                source=export_script_path,
                destination="/opt/ml/processing/input/scripts/",
                input_name="export_script"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="export_status",
                source="/opt/ml/processing/output",
                destination=f"{pipeline_prefix}/redshift-export-status",
            ),
        ],
        code=bootstrap_script_path,
        job_arguments=[
            "--host", redshift_host,
            "--database", redshift_database,
            "--user", redshift_user,
            "--cluster-identifier", redshift_cluster_id,
            "--iam-role", redshift_iam_role,
            "--s3-export-path", redshift_export_path,
            "--region", region,
            "--table", "dth_churn_ml_training.training_features"
            # Removed --sample-ratio for production: trains on full 50M row dataset
        ]
    )

    # =============================================================================
    # Step 1: Model Training (same as scheduled_training_large.py)
    # =============================================================================

    # Hyperparameters (exactly like the working script)
    hyperparams = {
        "mlflow-mode": "disabled",
        "config": "config.yaml",
        "chunk-size": "100000"
    }

    # Create PyTorch estimator (exactly like the working script)
    training_source_dir = os.path.join(os.path.dirname(__file__), "..", "training")
    pytorch_estimator = PyTorch(
        entry_point="train_sagemaker_large.py",
        source_dir=training_source_dir,
        role=role,
        instance_type=training_instance_type,
        instance_count=1,
        framework_version="2.0.0",
        py_version="py310",
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparams,
        max_run=10800,
        use_spot_instances=False,
        base_job_name=f"{pipeline_name}-training",
        output_path=f"{pipeline_prefix}/training-output",
        metric_definitions=[
            {
                'Name': 'roc_auc',
                'Regex': r'Final ROC-AUC: ([0-9\.]+)'
            },
            {
                'Name': 'pr_auc',
                'Regex': r'PR-AUC:\s+([0-9\.]+)'
            },
            {
                'Name': 'f1_score',
                'Regex': r'Final F1 @ target recall: ([0-9\.]+)'
            },
            {
                'Name': 'threshold',
                'Regex': r'Threshold:\s+([0-9\.]+)'
            },
            {
                'Name': 'churner_recall',
                'Regex': r'Churner recall:\s+([0-9\.]+)'
            },
            {
                'Name': 'churner_precision',
                'Regex': r'Churner precision:\s+([0-9\.]+)'
            },
            {
                'Name': 'accuracy',
                'Regex': r'Accuracy:\s+([0-9\.]+)'
            },
            {
                'Name': 'actual_churn_rate',
                'Regex': r'Total churners: ([0-9\.]+)%'
            },
            {
                'Name': 'training_samples',
                'Regex': r'Train samples: ([0-9,]+)'
            },
            {
                'Name': 'peak_memory_mb',
                'Regex': r'Peak memory usage: ([0-9\.]+) MB'
            },
            {
                'Name': 'final_data_shape_rows',
                'Regex': r'Final loaded data shape: \\(([0-9,]+),'
            }
        ],
        dependencies=[
            os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        ]
    )

    # Training input - use Redshift export path
    training_input = TrainingInput(
        s3_data=redshift_export_path,
        content_type="application/x-parquet"
    )

    # Training step - depends on Redshift export
    step_train = TrainingStep(
        name="ChurnTraining",
        estimator=pytorch_estimator,
        inputs={'training': training_input},
        depends_on=[step_redshift_export]
    )

    # =============================================================================
    # Step 2: Simple Evaluation
    # =============================================================================

    # Use a simple sklearn processor for evaluation
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{pipeline_name}-evaluation",
    )

    # Simple approach - check actual training job metrics to get the job name pattern
    # Then hardcode the extraction logic based on the model artifact path
    evaluation_script = f"""
import os
import json
import boto3

# Get training job metrics from SageMaker
sagemaker_client = boto3.client('sagemaker', region_name='{region}')

print("Looking for training job name from model artifacts...")

# Get model artifact path from processing input
model_path = "/opt/ml/processing/model"
model_files = os.listdir(model_path)
print(f"Model files: {{model_files}}")

# Find the tar.gz file
tar_file = None
for file in model_files:
    if file.endswith('.tar.gz'):
        tar_file = file
        break

if tar_file:
    print(f"Found model tar file: {{tar_file}}")

    # The training job name is in the S3 path, not the tar filename
    # We need to extract it from the S3 path structure
    # Pattern: s3://bucket/pipeline-name/training-output/TRAINING-JOB-NAME/output/model.tar.gz

    # For now, let's try to infer from current pipeline execution
    # Check if we can find training jobs that match our pipeline pattern

    try:
        # List recent training jobs and find the one that matches our pipeline pattern
        response = sagemaker_client.list_training_jobs(
            MaxResults=10,
            SortBy='CreationTime',
            SortOrder='Descending'
        )

        training_job_name = None
        for job in response.get('TrainingJobSummaries', []):
            job_name = job['TrainingJobName']
            print(f"Checking training job: {{job_name}}")

            # Look for pipeline training jobs
            if 'pipelines-' in job_name and 'ChurnTraining' in job_name:
                # Check if this job finished recently (within last hour)
                job_end_time = job.get('TrainingEndTime')
                if job_end_time:
                    import datetime
                    now = datetime.datetime.now(job_end_time.tzinfo)
                    time_diff = now - job_end_time
                    if time_diff.total_seconds() < 3600:  # Within last hour
                        training_job_name = job_name
                        print(f"Found recent pipeline training job: {{training_job_name}}")
                        break

        if training_job_name:
            # Get actual training job metrics
            response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
            training_metrics = response.get('FinalMetricDataList', [])

            # Convert to our evaluation format
            metrics = {{"regression_metrics": {{}}}}

            for metric in training_metrics:
                metric_name = metric['MetricName']
                metric_value = metric['Value']
                print(f"Found metric: {{metric_name}} = {{metric_value}}")

                metrics["regression_metrics"][metric_name] = {{"value": metric_value}}

            # Ensure we have all required metrics for conditional registration
            required_metrics = ['roc_auc', 'pr_auc', 'f1_score', 'churner_recall']
            for req_metric in required_metrics:
                if req_metric not in metrics["regression_metrics"]:
                    print(f"Warning: {{req_metric}} not found, setting to 0.0")
                    metrics["regression_metrics"][req_metric] = {{"value": 0.0}}

            print(f"Final extracted metrics: {{metrics}}")
        else:
            raise Exception("Could not find recent pipeline training job")

    except Exception as e:
        print(f"Error getting training metrics: {{e}}")
        print("Using fallback metrics that will fail quality gates")
        # Fallback metrics (intentionally low to fail quality gates)
        metrics = {{
            "regression_metrics": {{
                "roc_auc": {{"value": 0.5}},
                "pr_auc": {{"value": 0.1}},
                "f1_score": {{"value": 0.3}},
                "churner_recall": {{"value": 0.5}}
            }}
        }}
else:
    print("No model tar file found")
    metrics = {{
        "regression_metrics": {{
            "roc_auc": {{"value": 0.5}},
            "pr_auc": {{"value": 0.1}},
            "f1_score": {{"value": 0.3}},
            "churner_recall": {{"value": 0.5}}
        }}
    }}

# Create output directory
os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)

# Save evaluation results
with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Evaluation complete")
"""

    # Create a simple evaluation file locally
    os.makedirs("/tmp/pipeline", exist_ok=True)
    eval_script_path = "/tmp/pipeline/simple_evaluation.py"
    with open(eval_script_path, 'w') as f:
        f.write(evaluation_script)

    # Evaluation step
    step_eval = ProcessingStep(
        name="ChurnEvaluation",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"{pipeline_prefix}/evaluation",
            ),
        ],
        code=eval_script_path,
    )

    # =============================================================================
    # Step 3: Conditional Model Registration
    # =============================================================================

    # Model metrics
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri, "evaluation.json"]
            ),
            content_type="application/json"
        )
    )

    # Condition for model registration
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    step_eval.property_files = [evaluation_report]

    # Multiple quality conditions (ALL must pass)
    cond_gte_roc_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.roc_auc.value"
        ),
        right=roc_auc_threshold,
    )

    cond_gte_pr_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.pr_auc.value"
        ),
        right=pr_auc_threshold,
    )

    cond_gte_f1 = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.f1_score.value"
        ),
        right=f1_threshold,
    )

    cond_gte_churner_recall = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.churner_recall.value"
        ),
        right=churner_recall_threshold,
    )

    # Model registration
    step_register = RegisterModel(
        name="ChurnModelRegister",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # Conditional registration - ALL conditions must pass
    step_cond = ConditionStep(
        name="CheckModelQuality",
        conditions=[cond_gte_roc_auc, cond_gte_pr_auc, cond_gte_f1, cond_gte_churner_recall],
        if_steps=[step_register],
        else_steps=[],
    )

    # =============================================================================
    # Create Pipeline
    # =============================================================================

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            training_instance_type,
            roc_auc_threshold,
            pr_auc_threshold,
            f1_threshold,
            churner_recall_threshold,
        ],
        steps=[step_redshift_export, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple churn model pipeline")
    parser.add_argument("--create", action="store_true", help="Create the pipeline")
    parser.add_argument("--execute", action="store_true", help="Execute the pipeline")
    parser.add_argument("--pipeline-name", default="churn-training-pipeline-v2", help="Pipeline name")
    parser.add_argument("--model-package-group-name", default="churn-model-package-group-v2", help="Model package group name")

    args = parser.parse_args()

    if args.create or args.execute:
        print(f"🔧 Creating Simple SageMaker Pipeline: {args.pipeline_name}")

        pipeline = create_simple_churn_pipeline(
            pipeline_name=args.pipeline_name,
            model_package_group_name=args.model_package_group_name,
        )

        if args.create:
            pipeline.upsert(role_arn=sagemaker.get_execution_role())
            print(f"✅ Pipeline created/updated: {args.pipeline_name}")

        if args.execute:
            execution = pipeline.start()
            print(f"🚀 Pipeline execution started: {execution.arn}")

    else:
        print("📖 Simple SageMaker Churn Pipeline")
        print("Following the working scheduled_training_large.py pattern")
        print("\nUsage:")
        print("  --create    : Create/update the pipeline")
        print("  --execute   : Execute the pipeline")
        print("  --create --execute : Create and execute")


if __name__ == "__main__":
    main()
