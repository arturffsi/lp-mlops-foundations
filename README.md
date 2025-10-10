# ZAP Learning Pod: MLOps Foundations

## 🎯 Objectives
- Align ML and DevOps teams on shared MLOps practices and concepts.  
- Enable both teams to collaborate effectively across the full ML lifecycle.  
- Gain hands-on experience with MLOps tools and workflows, both locally and on AWS SageMaker with Redshift.  
- Remain model-agnostic: participants can choose their preferred ML models, applying best practices consistently.  
- Achieve **MLOps maturity**, where CI/CD automation supports reliable and scalable ML pipelines.  

---

## 🛠️ Prerequisites
- **Technical Skills**:  
  - Python (basic proficiency)  
  - SQL (basic proficiency)  
- **Tools & Accounts**:  
  - GitHub account (for version control and CI/CD)  
  - AWS account with access to SageMaker and Redshift  
- **Setup**:  
  - Python 3.9+ environment (conda or venv recommended)  
  - Jupyter Notebook / JupyterLab installed  
  - Recommended packages: `pandas`, `scikit-learn`, `mlflow`, `boto3`, `pytest`, `sagemaker`

---

## 🤔 Why MLOps Exists and Its Goals
Machine Learning (ML) on its own is not enough to deliver business value reliably — the challenge is not just building a good model, but **operating it continuously in production**.  

MLOps applies **DevOps principles to ML systems**, with the following goals:
- **Unify development and operations**: Bridge the gap between data scientists/ML engineers (focused on models) and DevOps engineers (focused on reliability).  
- **Automate the ML lifecycle**: Integration, testing, deployment, retraining, and monitoring are automated to reduce manual effort and human error.  
- **Enable reproducibility**: Ensure that models, data, and experiments can be consistently recreated.  
- **Increase velocity and reliability**: Allow teams to quickly try new ideas, push updates safely, and adapt to changing data/business needs.  
- **Monitor and maintain models**: Detect data drift, model decay, or infrastructure issues early to maintain model performance over time.  

In short: **MLOps exists to make ML systems reliable, scalable, and sustainable in production.**

---

## 🏁 ZAP’s Target: MLOps Level 2 Maturity
After completing this LP, ZAP aims to reach **MLOps Level 2 maturity** (as defined by Google).  

This means:
- ML pipelines are **automated** and integrated with **CI/CD workflows**.  
- Teams can **rapidly test, validate, and deploy** new data pipelines, features, and models.  
- The process includes:
  - Source control for all code and pipeline steps.  
  - Automated building, testing, and packaging of pipeline components.  
  - Deployment to different environments (dev → pre-prod → prod) with increasing levels of automation.  
  - Model registry and metadata tracking for reproducibility.  
  - Monitoring of deployed services for performance and reliability.  
- Retraining and redeployment of models can be **triggered automatically** (based on schedule, new data, or business needs).  

**Success for ZAP** means that ML and DevOps teams:  
- Collaboratively own **a shared, automated ML pipeline**.  
- Confidently manage the **full ML lifecycle**: data preparation, training, experiment tracking, testing, deployment, and monitoring.  
- Continuously deliver **reliable ML models** that adapt to changes in data and business context.  
- Build **mutual trust** between ML and DevOps teams through peer validation, shared tools, and transparent workflows.  

👉 In other words, ZAP will move from “developing MVP models” to **“operating ML pipelines with CI/CD automation.”**

---

## 🖼️ Visual Overview

![CI/CD and Automated ML Pipeline](images/CI_CD_and_automated_ML_pipeline.svg)

This diagram illustrates the end-to-end automation of ML pipelines, integrating CI/CD practices for reliable and scalable model deployment.

---

## 📅 Weekly Schedule
### Week 1: Exploratory Data Analysis (EDA)  
- Import, clean, and explore datasets.  
- Understand data quality issues and preprocessing.  
- Exercises: perform EDA on Redshift sample dataset.  

### Week 2: Pipelines and ETL  
- Build ETL workflows with reproducible pipelines.  
- Discuss importance of consistent naming conventions at ZAP.  
- Exercises: implement an ETL pipeline feeding Redshift data into a local ML experiment.  

### Week 3: Experiment Tracking and Model Versioning (MLflow)  
- Track experiments, hyperparameters, and results.  
- Register and version models with MLflow.  
- Exercises: log training runs and compare metrics.  

### Week 4: CI/CD and Test Coverage  
- Apply CI/CD principles to ML pipelines.  
- Differentiate development and production environments.  
- Exercises: create tests for data preprocessing and model training.  

### Week 5: Deployment on AWS SageMaker  
- Deploy models as endpoints in SageMaker.  
- Integrate pipelines for automated deployment.  
- Exercises: deploy a trained model and test predictions.  

### Week 6: Continuous Monitoring  
- Monitor data drift, model drift, and system health.  
- Implement alerts and retraining triggers.  
- Exercises: simulate drift and design a retraining workflow.  

---

## ✅ Expected Outcomes
By completing this LP, participants will:  
- Understand and apply MLOps best practices end-to-end.  
- Use AWS SageMaker and Redshift in ML pipelines.  
- Track, version, and deploy models with reproducibility.  
- Collaborate across ML and DevOps boundaries with shared language and workflows.  
- Be prepared to discuss and refine ZAP’s internal MLOps ownership strategy.  
- Operate at **MLOps Level 2 maturity**, where CI/CD-automated ML pipelines ensure reliability and scalability.  

---

## 🤝 Peer Validation
- Each week ends with a quiz or coding challenge.  
- Participants must review at least one peer’s submission.  
- Feedback should cover correctness, clarity, and alignment with best practices.  
- Peer validation ensures shared understanding and fosters cross-team trust.  

---

## 📚 Additional Resources
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/index.html)  
- [Amazon Redshift Documentation](https://docs.aws.amazon.com/redshift/index.html)  
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)  
- [MLOps Principles by Google](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)  
