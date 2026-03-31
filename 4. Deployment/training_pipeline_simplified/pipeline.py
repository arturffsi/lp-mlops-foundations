"""
Simplified SageMaker Pipeline for Churn Prediction

This creates a complete SageMaker Pipeline with all steps:
1. Export data from Redshift to S3
2. Train model
3. Evaluate model
4. Conditionally register model (if metrics are good)

Much simpler than the production version, but demonstrates all key concepts.

Usage:
    python pipeline.py --create              # Create/update pipeline
    python pipeline.py --execute             # Execute pipeline
    python pipeline.py --create --execute    # Create and execute
"""

import argparse
import os
from datetime import datetime
import sagemaker
from sagemaker.pytorch import PyTorch                          # Estimator that runs train.py inside a PyTorch container
from sagemaker.sklearn.processing import SKLearnProcessor     # Lightweight processor for scikit-learn scripts (used for evaluation)
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput, NetworkConfig
from sagemaker.workflow.pipeline import Pipeline              # The pipeline object that holds all steps together
from sagemaker.workflow.steps import TrainingStep, ProcessingStep  # Wrappers that turn scripts into pipeline steps
from sagemaker.workflow.step_collections import RegisterModel # Registers the model in the SageMaker Model Registry
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat  # Pipeline parameters (overridable at runtime)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo  # Used to check if a metric passes a threshold
from sagemaker.workflow.condition_step import ConditionStep   # A step that branches: if condition passes → register, else → skip
from sagemaker.workflow.functions import JsonGet, Join        # JsonGet reads a value from a JSON file; Join builds S3 paths
from sagemaker.workflow.properties import PropertyFile        # Tells SageMaker to read an output file so conditions can use its values
from sagemaker.inputs import TrainingInput                    # Points the training step to the S3 data location
from sagemaker.model_metrics import MetricsSource, ModelMetrics  # Attaches evaluation metrics to the registered model
from sagemaker import image_uris                              # Looks up the right Docker image URI for a given framework/region


def create_pipeline(
    pipeline_name="learningpods-pipeline",
    model_package_group_name="learningpods-model-group",
    region="af-south-1"
):
    """
    Create a complete SageMaker Pipeline with all 4 steps.

    Args:
        pipeline_name: Name of the pipeline
        model_package_group_name: Model registry group name
        region: AWS region

    Returns:
        SageMaker Pipeline object
    """
    print(f"\n{'='*60}")
    print(f"🔧 Creating SageMaker Pipeline: {pipeline_name}")
    print(f"{'='*60}\n")

    # Get SageMaker session and role
    # Use boto3 session with the right profile for local execution
    import boto3
    try:
        boto_session = boto3.Session(profile_name="SageMaker-Full-AI-Access-733246370304")
        print(f"✅ Using AWS profile: SageMaker-Full-AI-Access-733246370304")
    except Exception:
        boto_session = boto3.Session()
        print(f"✅ Using default AWS credentials")

    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    # Use a proper SageMaker execution role (not SSO assumed role)
    sts = boto_session.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/service-role/SageMaker-ExecutionRole-20250915T083600"
    print(f"✅ Using SageMaker execution role: {role}")

    bucket = sagemaker_session.default_bucket()

    # Create datestamp for this run
    datestamp = datetime.now().strftime("%Y-%m-%d")

    # S3 paths
    pipeline_prefix = f"s3://{bucket}/{pipeline_name}"
    redshift_export_path = f"s3://{bucket}/{pipeline_name}/redshift_export_{datestamp}/"
    # Data will come from Redshift export step
    data_path = redshift_export_path

    print(f"📋 Configuration:")
    print(f"   Pipeline: {pipeline_name}")
    print(f"   Region: {region}")
    print(f"   Role: {role}")
    print(f"   Bucket: {bucket}")
    print(f"   Datestamp: {datestamp}")
    print()

    # =========================================================================
    # Pipeline Parameters
    # =========================================================================
    # Parameters are like function arguments for the whole pipeline.
    # They have default values baked in, but can be overridden at execution time
    # without changing or re-uploading the pipeline definition.
    # e.g. python pipeline.py --execute --n-estimators 1000 --learning-rate 0.05

    instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.2xlarge"  # Using ml.m5.2xlarge (has quota in this account)
    )

    n_estimators = ParameterInteger(
        name="NEstimators",
        default_value=811
    )

    learning_rate = ParameterFloat(
        name="LearningRate",
        default_value=0.044
    )

    depth = ParameterInteger(
        name="Depth",
        default_value=5
    )

    l2_leaf_reg = ParameterFloat(
        name="L2LeafReg",
        default_value=1.015
    )

    # Quality gate thresholds — the model will only be registered if ALL of these pass.
    # These are also pipeline parameters, so you can lower/raise them at execution time.
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
        default_value=0.5
    )

    recall_threshold = ParameterFloat(
        name="RecallThreshold",
        default_value=0.7
    )

    print(f"⚙️  Pipeline Parameters:")
    print(f"   Training Instance: {instance_type.default_value}")
    print(f"   Hyperparameters:")
    print(f"     - n_estimators: {n_estimators.default_value}")
    print(f"     - learning_rate: {learning_rate.default_value}")
    print(f"     - depth: {depth.default_value}")
    print(f"     - l2_leaf_reg: {l2_leaf_reg.default_value}")
    print(f"   Quality Gates:")
    print(f"     - ROC-AUC >= {roc_auc_threshold.default_value}")
    print(f"     - PR-AUC >= {pr_auc_threshold.default_value}")
    print(f"     - F1 Score >= {f1_threshold.default_value}")
    print(f"     - Recall >= {recall_threshold.default_value}")
    print()

    # =========================================================================
    # Step 1: Export Data from Redshift to S3
    # =========================================================================

    print(f"🔧 Configuring Step 1: Redshift Export...")

    # Get PyTorch image for processing
    pytorch_image = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.0.0",
        py_version="py310",
        instance_type="ml.m5.large",
        image_scope="training"
    )

    # VPC config - same as production pipeline (churn-training-pipeline-v2)
    # Redshift lives inside a private VPC (not exposed to the internet).
    # Without this, the SageMaker container can't reach it.
    # The subnet and security group IDs are the same ones used in production.
    vpc_config = NetworkConfig(
        security_group_ids=["sg-017fb44c1ff043deb"],
        subnets=["subnet-06cf22533465bffc6", "subnet-04a897538d1ac1390"],
        enable_network_isolation=False
    )

    # A ScriptProcessor runs any Python script inside a Docker container on SageMaker.
    # Here we reuse the PyTorch image (it already has the psycopg2/boto3 dependencies we need).
    # Think of it as: "spin up a machine, run export_redshift.py, then shut it down."
    redshift_processor = ScriptProcessor(
        image_uri=pytorch_image,
        command=["python3"],
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{pipeline_name}-redshift-export",
        network_config=vpc_config
    )

    # Redshift needs its own IAM role to write data to S3 (the UNLOAD command requires it).
    # This is separate from the SageMaker execution role — Redshift uses it to auth with S3.
    redshift_iam_role = f"arn:aws:iam::{account_id}:role/RedshiftIAMAuthRole"

    # Redshift export step: runs export_redshift.py inside the container.
    # job_arguments are passed as CLI args to the script (like running it on the terminal).
    # --modulo-divisor 333 means: only export rows where idconsumo % 333 = 0 (~0.3% of data),
    # which is much faster than RANDOM() because Redshift can use the index.
    step_redshift_export = ProcessingStep(
        name="ExportFromRedshift",
        processor=redshift_processor,
        code="export_redshift.py",
        job_arguments=[
            "--host", "zap-airflow-endpoint-omdjytkjkmzjhjkmiaoe.cl4o4mmtx9ir.af-south-1.redshift.amazonaws.com",
            "--database", "prod",
            "--user", "svc_sagemaker",
            "--cluster-identifier", "redshift-cluster-dsi",
            "--iam-role", redshift_iam_role,
            "--table", "dth_churn_ml_training.training_features",
            "--modulo-divisor", "333",  # ~0.3% sample, much faster than RANDOM()
            "--output-path", redshift_export_path,
            "--region", region
        ],
        outputs=[
            ProcessingOutput(
                output_name="export_status",
                source="/opt/ml/processing/output",
                destination=f"{pipeline_prefix}/export-status"
            )
        ]
    )

    print(f"   ✅ Redshift export configured")

    # =========================================================================
    # Step 2: Train Model
    # =========================================================================

    print(f"🔧 Configuring Step 2: Model Training...")

    # A PyTorch estimator tells SageMaker how to run train.py.
    # SageMaker will: spin up an ml.m5.2xlarge, copy the source_dir code into it,
    # install requirements.txt, and run train.py with the given hyperparameters.
    # metric_definitions scrape stdout with regex so SageMaker can track metrics.
    pytorch_estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version='2.0.0',
        py_version='py310',
        sagemaker_session=sagemaker_session,
        hyperparameters={
            'n-estimators': n_estimators,
            'learning-rate': learning_rate,
            'depth': depth,
            'l2-leaf-reg': l2_leaf_reg
        },
        base_job_name=f"{pipeline_name}-training",
        output_path=f"{pipeline_prefix}/training-output",
        metric_definitions=[
            {'Name': 'roc_auc', 'Regex': r'ROC-AUC:\s+([0-9\.]+)'},
            {'Name': 'pr_auc', 'Regex': r'PR-AUC:\s+([0-9\.]+)'},
            {'Name': 'f1_score', 'Regex': r'F1 Score:\s+([0-9\.]+)'},
            {'Name': 'recall', 'Regex': r'Actual Recall:\s+([0-9\.]+)%'},
            {'Name': 'precision', 'Regex': r'Precision:\s+([0-9\.]+)%'},
        ]
    )

    # TrainingStep wraps the estimator into a pipeline step.
    # depends_on=[step_redshift_export] ensures this step only starts after the export finishes.
    # The s3_data points to the same S3 path where export_redshift.py wrote the parquet files.
    step_train = TrainingStep(
        name="TrainChurnModel",
        estimator=pytorch_estimator,
        inputs={
            'training': TrainingInput(
                s3_data=data_path,
                content_type="application/x-parquet"
            )
        },
        depends_on=[step_redshift_export]
    )

    print(f"   ✅ Training configured")

    # =========================================================================
    # Step 3: Evaluate Model
    # =========================================================================

    print(f"🔧 Configuring Step 3: Model Evaluation...")

    # SKLearnProcessor is a lightweight processor pre-loaded with scikit-learn.
    # We use it here just to run evaluate_model.py — it doesn't need a GPU or PyTorch.
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        base_job_name=f"{pipeline_name}-evaluation"
    )

    # Evaluation step: runs evaluate_model.py.
    # inputs: the model artifact (model.tar.gz) from the training step is downloaded
    #         into /opt/ml/processing/model inside the container.
    # outputs: evaluate_model.py writes evaluation.json to /opt/ml/processing/evaluation,
    #          which SageMaker then uploads to S3. The next step reads it from there.
    # step_train.properties.TrainingJobName is resolved at runtime — SageMaker fills in
    # the actual job name when the pipeline executes.
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=sklearn_processor,
        code="evaluate_model.py",
        job_arguments=[
            "--region", region,
            "--training-job-name", step_train.properties.TrainingJobName,
        ],
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"{pipeline_prefix}/evaluation"
            )
        ]
    )

    print(f"   ✅ Evaluation configured")

    # =========================================================================
    # Step 4: Conditional Model Registration
    # =========================================================================

    print(f"🔧 Configuring Step 4: Conditional Registration...")

    # Attach the evaluation.json to the registered model so it's visible in the Model Registry UI.
    # Join() builds the S3 path: <evaluation output S3 URI>/evaluation.json
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[step_eval.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri, "evaluation.json"]
            ),
            content_type="application/json"
        )
    )

    # PropertyFile tells SageMaker to read evaluation.json from the "evaluation" output.
    # This is what makes it possible to use JsonGet() in conditions below —
    # without this, the pipeline wouldn't know where to find the metric values.
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    step_eval.property_files = [evaluation_report]

    # Quality gate conditions: each one reads a metric from evaluation.json and
    # compares it against the threshold parameter.
    # JsonGet navigates the JSON like: evaluation.json → regression_metrics → roc_auc → value
    # ALL conditions must be True for the model to be registered (AND logic).
    cond_roc_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.roc_auc.value"
        ),
        right=roc_auc_threshold
    )

    cond_pr_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.pr_auc.value"
        ),
        right=pr_auc_threshold
    )

    cond_f1 = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.f1_score.value"
        ),
        right=f1_threshold
    )

    cond_recall = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.recall.value"
        ),
        right=recall_threshold
    )

    # RegisterModel saves the model to the SageMaker Model Registry.
    # approval_status="PendingManualApproval" means a human must approve it before
    # it can be deployed. This is a safety gate before production.
    step_register = RegisterModel(
        name="RegisterChurnModel",
        estimator=pytorch_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics
    )

    # ConditionStep is the branching logic of the pipeline:
    # - if ALL conditions pass → run if_steps (register the model)
    # - else → run else_steps (empty = do nothing, pipeline ends without registering)
    step_cond = ConditionStep(
        name="CheckModelQuality",
        conditions=[cond_roc_auc, cond_pr_auc, cond_f1, cond_recall],
        if_steps=[step_register],
        else_steps=[]
    )

    print(f"   ✅ Conditional registration configured")
    print(f"   📊 Model will register ONLY if:")
    print(f"      - ROC-AUC >= {roc_auc_threshold.default_value}")
    print(f"      - PR-AUC >= {pr_auc_threshold.default_value}")
    print(f"      - F1 Score >= {f1_threshold.default_value}")
    print(f"      - Recall >= {recall_threshold.default_value}")

    # =========================================================================
    # Create Pipeline
    # =========================================================================

    print(f"\n🔨 Assembling pipeline...")

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_type,
            n_estimators,
            learning_rate,
            depth,
            l2_leaf_reg,
            roc_auc_threshold,
            pr_auc_threshold,
            f1_threshold,
            recall_threshold
        ],
        steps=[
            step_redshift_export,
            step_train,
            step_eval,
            step_cond
        ],
        sagemaker_session=sagemaker_session
    )

    print(f"✅ Pipeline created with 4 steps:")
    print(f"   1. ExportFromRedshift (export data to S3)")
    print(f"   2. TrainChurnModel")
    print(f"   3. EvaluateModel")
    print(f"   4. CheckModelQuality (conditional registration)")
    print(f"{'='*60}\n")

    return pipeline, role


def main():
    parser = argparse.ArgumentParser(description="Simplified SageMaker Pipeline for churn prediction")

    parser.add_argument("--create", action="store_true",
                       help="Create/update the pipeline")
    parser.add_argument("--execute", action="store_true",
                       help="Execute the pipeline")
    parser.add_argument("--pipeline-name", default="learningpods-pipeline",
                       help="Name of the pipeline")
    parser.add_argument("--model-group", default="learningpods-model-group",
                       help="Model package group name")
    parser.add_argument("--region", default="af-south-1",
                       help="AWS region")

    # Parameter overrides
    parser.add_argument("--instance-type", default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--l2-leaf-reg", type=float, default=None)
    parser.add_argument("--roc-auc-threshold", type=float, default=None)
    parser.add_argument("--pr-auc-threshold", type=float, default=None)
    parser.add_argument("--f1-threshold", type=float, default=None)
    parser.add_argument("--recall-threshold", type=float, default=None)

    args = parser.parse_args()

    if not args.create and not args.execute:
        print("\n" + "="*60)
        print("📖 Simplified SageMaker Pipeline for Churn Prediction")
        print("="*60)
        print("\n🎯 Pipeline Steps:")
        print("   1. ExportFromRedshift   - Export fresh data from Redshift to S3")
        print("   2. TrainChurnModel      - Train CatBoost model")
        print("   3. EvaluateModel        - Extract metrics from training")
        print("   4. CheckModelQuality    - Conditionally register if metrics pass")
        print("\nUsage:")
        print("  python pipeline.py --create              # Create/update pipeline")
        print("  python pipeline.py --execute             # Execute pipeline")
        print("  python pipeline.py --create --execute    # Create and execute")
        print("\nExamples:")
        print("  # Run with default parameters")
        print("  python pipeline.py --create --execute")
        print()
        print("  # Custom hyperparameters")
        print("  python pipeline.py --execute --n-estimators 1000 --learning-rate 0.05")
        print()
        print("  # Custom quality gates")
        print("  python pipeline.py --execute --roc-auc-threshold 0.75 --recall-threshold 0.8")
        print("="*60 + "\n")
        return

    # Create pipeline
    pipeline, role = create_pipeline(
        pipeline_name=args.pipeline_name,
        model_package_group_name=args.model_group,
        region=args.region
    )

    # Create/update pipeline
    if args.create:
        print(f"📤 Uploading pipeline definition to SageMaker...")
        # upsert = create if it doesn't exist, update if it does.
        # Safe to run multiple times — it won't delete past runs or registered models.
        pipeline.upsert(role_arn=role)
        print(f"✅ Pipeline '{args.pipeline_name}' created/updated!\n")

    # Execute pipeline
    if args.execute:
        print(f"🚀 Starting pipeline execution...")

        # Build execution parameters
        execution_params = {}
        if args.instance_type:
            execution_params['TrainingInstanceType'] = args.instance_type
        if args.n_estimators:
            execution_params['NEstimators'] = args.n_estimators
        if args.learning_rate:
            execution_params['LearningRate'] = args.learning_rate
        if args.depth:
            execution_params['Depth'] = args.depth
        if args.l2_leaf_reg:
            execution_params['L2LeafReg'] = args.l2_leaf_reg
        if args.roc_auc_threshold:
            execution_params['RocAucThreshold'] = args.roc_auc_threshold
        if args.pr_auc_threshold:
            execution_params['PrAucThreshold'] = args.pr_auc_threshold
        if args.f1_threshold:
            execution_params['F1Threshold'] = args.f1_threshold
        if args.recall_threshold:
            execution_params['RecallThreshold'] = args.recall_threshold

        if execution_params:
            print(f"   Custom parameters: {execution_params}")
            execution = pipeline.start(parameters=execution_params)
        else:
            print(f"   Using default parameters")
            execution = pipeline.start()

        print(f"\n✅ Pipeline execution started!")
        print(f"   Execution ARN: {execution.arn}")
        print(f"\n📊 Monitor progress at:")
        print(f"   {f'https://console.aws.amazon.com/sagemaker/home?region={args.region}#/pipelines'}")
        print()


if __name__ == "__main__":
    main()
