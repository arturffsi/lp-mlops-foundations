# 🤖 Week 2 — Model Training: Objectives and Overview

> **Goal:** Develop and refine a machine learning model using reproducible, version-controlled, and automated practices aligned with MLOps Level 2 maturity.

## Quick start

```bash
cd "2. Training"
pip install -r requirements.txt
```

**Run order matters** — each notebook reads the previous one's output:

1. **`../1. EDA/eda_example.ipynb`** — must be run first; produces `eda_example_results/{train,val,test}.parquet`.
2. **`1_data_preparation_example.ipynb`** → `data_preparation_output/`
3. **`2_model_selection_example.ipynb`** → `model_selection_output/`
4. **`3_experimentation_hyperparameter_example.ipynb`** → `hyperparameter_tuning_output/`

If you skip step 1, the data-preparation notebook will raise `FileNotFoundError` on the EDA splits.

---

## 🎯 Learning Objectives

By the end of this week, participants will:
- Understand the **MLOps training step** and its role in the end-to-end ML lifecycle.
- Prepare and **version training data** to ensure reproducibility and traceability.
- Perform **feature engineering** and document transformations for consistency across environments.
- Conduct **model experimentation**, comparing algorithms and configurations systematically.
- Apply **hyperparameter tuning** with Optuna and understand when it's worth the computational cost.
- Validate and select models based on **performance metrics** and **business relevance**.
- Master **threshold tuning** to meet business requirements (e.g., achieving target recall for imbalanced datasets).

---

## ⚙️ The MLOps Training Step

The **MLOps training step** is the process of developing and refining a machine learning model using data and algorithms in a **controlled, reproducible, and automated way**.

It bridges **data preparation** and **model deployment**, ensuring that the model you train can be reliably reproduced, validated, and deployed through CI/CD workflows.

This phase typically includes:
1. Preparing and versioning datasets.
2. Experimenting with models and hyperparameters.
3. Tracking all experiments, metadata, and results.
4. Validating model performance.
5. Promoting the best model to the deployment stage.

---

## 🔑 Key Activities in the Model Training Step

### 1️⃣ Data Preparation and Feature Engineering
- Clean, preprocess, and transform raw data into a **training-ready format**.
- Handle missing values, encode categorical variables, and scale numerical features.
- Create and document **engineered features** that improve model learning.
- Store and version both the **transformation code** and **processed datasets** for reproducibility.

**💡 Key Teaching Point:** While CatBoost/XGBoost handle missing values natively, we teach explicit filling because:
- Works with ANY model (not just tree-based)
- Gives explicit control over preprocessing strategy
- Maintains production consistency across the ML pipeline
- Better for understanding and documenting data quality issues

> ✅ *Outcome:* A reproducible dataset and feature set ready for model training.

---

### 2️⃣ Model Selection
- Choose the most appropriate **machine learning algorithm** for the problem type:
  - **Supervised Learning:** Classification, regression
  - **Unsupervised Learning:** Clustering, dimensionality reduction
  - **Reinforcement Learning:** Sequential decision-making
- Justify model selection based on data characteristics, business objectives, and interpretability needs.

> ✅ *Outcome:* A shortlist of candidate models suitable for experimentation.

---

### 3️⃣ Experimentation and Hyperparameter Tuning
- Train models using different **hyperparameter configurations** (e.g., learning rate, tree depth, regularization).
- Use **experiment tracking** (e.g., Optuna, MLflow, SageMaker Experiments) to log:
  - Model parameters and metrics
  - Training duration and hardware configuration
  - Data version and code commit hash
- Compare runs and visualize performance metrics to identify the best configuration.
- Document performance trade-offs (accuracy, latency, cost, interpretability).

**Key Skills Taught:**
- Defining search spaces with Optuna (suggest_int, suggest_float)
- Implementing objective functions for optimization
- Using TPESampler for intelligent hyperparameter search
- Visualizing hyperparameter importance and optimization history
- Understanding when tuning provides sufficient ROI

> ✅ *Outcome:* A reproducible record of experiments and a validated model candidate.

---

## 📓 Training Notebooks

This week includes three main workflows, each with **example** and **exercise** notebooks:

```
📊 EDA outputs (eda_example_results/)
    ↓
🧹 1. Data Preparation → data_preparation_output/
    ↓
🤖 2. Model Selection → model_selection_output/
    ↓
🔬 3. Hyperparameter Tuning → hyperparameter_tuning_output/
```

### 📚 Example vs Exercise Notebooks

Each workflow has **TWO notebooks**:

#### 📖 Example Notebooks (`*_example.ipynb`)
- Complete walkthrough with detailed explanations
- All code executed with outputs visible
- Reference material for learning concepts
- Production-grade practices demonstrated

#### ✏️ Exercise Notebooks (`*_exercise.ipynb`)
- Hands-on practice with guided tasks
- Students fill in code marked with `# YOUR CODE HERE`
- Includes discussion questions (Why? What if?)
- Reflection questions to promote deeper understanding
- Clear grading rubric (100 points each)
- Submission checklist for self-assessment
- **Estimated time:** 45-90 minutes per exercise

**💡 Recommended Approach:**
1. Study the example notebook first to understand concepts
2. Close the example notebook (resist peeking!)
3. Complete the exercise notebook independently
4. Compare your solution with the example afterward

---

### 1. Data Preparation (`1_data_preparation_*.ipynb`)

**Purpose:** Clean and prepare data from EDA outputs for model training.

**What you'll learn:**
- Loading EDA outputs (train/val/test splits)
- Identifying feature types (integer, numeric, categorical, datetime)
- Handling missing values with production strategies:
  - Integer features → -1
  - Numeric features → 0.0
  - Categorical features → '<MISSING>'
- Why we apply the SAME cleaning function to train/val/test (prevent leakage)
- Data validation before training
- Saving cleaned data for reproducibility

**Example notebook:** Complete walkthrough of data cleaning pipeline
**Exercise notebook:** 5 exercises (100 points)
- Exercise 1: Load data (15 pts)
- Exercise 2: Identify feature types (20 pts)
- Exercise 3: Create cleaning function (30 pts) ⭐ Core skill
- Exercise 4: Validate cleaned data (20 pts)
- Exercise 5: Save cleaned data (15 pts)

**Outputs:** `data_preparation_output/` with cleaned train/val/test parquet files

**Time estimate:** 45-60 minutes

---

### 2. Model Selection (`2_model_selection_*.ipynb`)

**Purpose:** Train and compare models, tune thresholds for business goals.

**What you'll learn:**
- Training CatBoost and XGBoost models
- Understanding CatBoost Pools and categorical feature handling
- Calculating evaluation metrics (ROC-AUC, PR-AUC, F1)
- Understanding the **Precision-Recall tradeoff**
- **Threshold tuning** to achieve target recall (80%)
- Comparing multiple models systematically
- Feature importance analysis
- Saving models and results for deployment

#### 🎯 Critical Skill: Threshold Tuning

One of the most important real-world skills taught in this module is **threshold tuning**:

**The Problem:**
- Default threshold (0.5) doesn't give control over recall/precision
- For imbalanced datasets, we often need to catch a specific % of positive cases
- Example: Catch 80% of churners, even if it means more false alarms

**The Solution:**
- Get probability predictions (not hard 0/1 classifications)
- Use `precision_recall_curve()` to find threshold for target recall
- Understand the tradeoff: Lower threshold = Higher recall, Lower precision

**Business Context:**
- Understand cost of false positives vs false negatives
- Make data-driven decisions aligned with business goals
- This skill is often overlooked in academic ML courses but critical in production!

**Example notebook:** Complete model training with threshold tuning
**Exercise notebook:** 5 exercises (100 points)
- Exercise 1: Load and prepare data (15 pts)
- Exercise 2: Train CatBoost model (25 pts)
- Exercise 3: Evaluate and tune threshold (30 pts) ⭐ Core skill
- Exercise 4: Visualize threshold impact (15 pts)
- Exercise 5: Save model and results (15 pts)

**Outputs:** `model_selection_output/` with best model, results, and feature importance

**Time estimate:** 60-75 minutes

---

### 3. Hyperparameter Tuning (`3_experimentation_hyperparameter_*.ipynb`)

**Purpose:** Automatically find optimal hyperparameters using Optuna.

**What you'll learn:**
- What hyperparameters are and why they differ from model parameters
- How hyperparameters control model learning (iterations, learning_rate, depth, regularization)
- Using Optuna's TPESampler for intelligent search
- Defining search spaces (suggest_int, suggest_float with log scale)
- Implementing objective functions that Optuna optimizes
- Tracking and visualizing all experiment trials
- Analyzing hyperparameter importance
- Understanding optimization history
- Comparing baseline vs tuned model performance
- Knowing when hyperparameter tuning is worth the cost

**Key Insights:**
- Sometimes baseline models are already good (realistic expectations!)
- More trials = better results, but with diminishing returns
- Always optimize on validation set (not training) to prevent overfitting
- Document all trials for reproducibility and learning

**Example notebook:** Complete hyperparameter tuning workflow with Optuna
**Exercise notebook:** 5 exercises (100 points)
- Exercise 1: Load data and train baseline (20 pts)
- Exercise 2: Define Optuna objective function (25 pts) ⭐ Core skill
- Exercise 3: Run hyperparameter tuning (25 pts)
- Exercise 4: Visualize optimization results (15 pts)
- Exercise 5: Train and save final model (15 pts)

**Outputs:** `hyperparameter_tuning_output/` with tuned model, best hyperparameters, and all trial results

**Time estimate:** 75-90 minutes

---

## 📦 Expected Outputs

By the end of this week, participants should produce:

### From Example Notebooks:

#### Data Preparation:
- `data_preparation_output/train.parquet` - Cleaned training data
- `data_preparation_output/val.parquet` - Cleaned validation data
- `data_preparation_output/test.parquet` - Cleaned test data
- `data_preparation_output/summary.json` - Cleaning summary

#### Model Selection:
- `model_selection_output/catboost_model.pkl` or `xgboost_model.pkl` - Trained model
- `model_selection_output/results.json` - Model metrics (ROC-AUC, PR-AUC, threshold, precision, recall, F1)
- `model_selection_output/feature_importance.csv` - Feature importance rankings

#### Hyperparameter Tuning:
- `hyperparameter_tuning_output/tuned_model.pkl` - Best tuned model
- `hyperparameter_tuning_output/best_hyperparameters.json` - Optimal hyperparameters
- `hyperparameter_tuning_output/results.json` - Performance metrics and baseline comparison
- `hyperparameter_tuning_output/all_trials.csv` - Complete experiment history
- `hyperparameter_tuning_output/feature_importance.csv` - Feature importance from best model

### From Exercise Notebooks:

All exercises produce the same outputs in `*_exercise_output/` directories, plus:
- Completed code cells demonstrating understanding
- Answers to discussion questions
- Answers to reflection questions
- Demonstrated ability to implement ML workflows independently

---

## 🎓 Pedagogical Approach

### Learning Methodology:
1. **Observe** - Study the example notebook (concepts, code, outputs)
2. **Practice** - Complete exercise notebook independently
3. **Reflect** - Answer discussion and reflection questions
4. **Verify** - Use submission checklist to ensure completion
5. **Compare** - Review your solution against the example

### Question Types:
- **Discussion Questions** - Test understanding of specific concepts
- **Reflection Questions** - Promote critical thinking about when/why to use techniques
- **Production Questions** - Connect learning to real-world ML pipelines

### Assessment:
- Each exercise is worth 100 points
- Points distributed across coding tasks and comprehension questions
- Self-assessment via submission checklists
- Instructor can review submissions and provide feedback

---

## 🔁 MLOps Integration

Training is not a one-time process — it's part of a **continuous cycle**:
- Automated retraining is triggered by **new data** or **drift detection**.
- All experiments are **tracked and versioned**, ensuring reproducibility.
- The **best-performing model** is seamlessly promoted to deployment pipelines.

This step ensures that models evolve reliably and remain aligned with changing data and business conditions.

**Key MLOps Principles Applied:**
- **Reproducibility** - Fixed seeds, versioned data, documented transformations
- **Traceability** - All experiments logged with metadata
- **Automation** - Optuna for automated hyperparameter search
- **Validation** - Rigorous data quality checks before training
- **Documentation** - JSON artifacts for downstream pipeline integration

---

## 🚀 Next Steps

After completing this week's notebooks, participants will be ready to:

1. **Deploy models to production** - Use trained models in serving infrastructure
2. **Monitor model performance** - Track metrics and detect drift
3. **Implement CI/CD for ML** - Automate the training → deployment pipeline
4. **Handle model versioning** - Manage multiple model versions in production
5. **Retrain systematically** - Trigger retraining based on performance degradation

---

> **Next Step → Deployment**
> Once a model is validated and versioned, the next stage focuses on **deploying** it in a scalable, monitored production environment using CI/CD automation.
