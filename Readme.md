# IEEE-CIS Fraud Detection: Production MLOps Pipeline 🕵️‍♂️

A modular, scalable, and **production-ready** machine learning pipeline designed for the Kaggle [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) competition.

This project moves beyond standard Jupyter Notebooks, implementing a robust engineering architecture that orchestrates the entire lifecycle: from PySpark ETL and Bayesian Hyperparameter Tuning to Ensemble Inference.

## 🏗 Project Architecture

The codebase follows a modular design, separating configuration, business logic, and execution control.

```text
ieee_fraud_detection/
├── data/                   # Data storage (Raw, Processed, Submissions) - Ignored in Git
├── scripts/                # CLI Entry Points
│   └── run_pipeline.py     # Main Orchestrator (The "Control Center")
├── src/                    # Core Logic (Python Package)
│   ├── config.py           # Typed Configuration (Pydantic)
│   ├── preprocess.py       # PySpark ETL (Train/Test consistency, Time-split)
│   ├── tuning.py           # Optuna Optimization Engine
│   ├── training.py         # Model Factory & Training (XGB/LGBM/Cat)
│   ├── ensemble.py         # Validation & Blending Logic
│   └── inference.py        # Final Submission Generation
├── .env                    # Secrets & MLflow Config (Not committed)
├── requirements.txt        # Project Dependencies
└── README.md               # Documentation
```

## 🛠 Tech Stack

* **ETL & Big Data:** **PySpark 3.x** (Handling large datasets, robust Feature Engineering, and categorical encoding).
* **Modeling:** **XGBoost**, **LightGBM**, **CatBoost** (GPU Accelerated).
* **Optimization:** **Optuna** (Automated Bayesian Hyperparameter Tuning).
* **MLOps:** **MLflow** (Experiment Tracking, Artifact Storage, and Model Registry).
* **Configuration:** **Pydantic** (Type-safe configuration and environment management).

## 🚀 Quick Start (How to Replicate)

Follow these steps to set up the project on a new machine.

### 1. Clone and Install

It is recommended to use a virtual environment with Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/your-username/ieee-fraud-detection.git
cd ieee-fraud-detection

# Create environment (Optional but recommended)
conda create -n fraud_detection python=3.10
conda activate fraud_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration (.env)

This project uses `python-dotenv` to manage secrets. You must create a `.env` file in the root directory to configure MLflow and Databricks access.

Create a file named `.env` and add the following credentials:

```ini
# .env file example

# MLflow Tracking URI (e.g., databricks, http://localhost:5000, or file:./mlruns)
MLFLOW_TRACKING_URI=databricks

# Databricks Credentials (Required if using Databricks for tracking)
DATABRICKS_HOST=https://<your-databricks-workspace-url>
DATABRICKS_TOKEN=dapi<your-personal-access-token>

# MLflow Experiment Path
MLFLOW_EXPERIMENT_NAME=/Shared/Fraud_detection
```

### 3. Data Setup

Download the competition datasets (`train_transaction.csv`, `train_identity.csv`, etc.) and place them in:
`data/raw/`

### 4. Pipeline Execution

The entire project is controlled via a single CLI entry point: `scripts/run_pipeline.py`.

#### 🔹 Step 1: Preprocessing (ETL)

Cleans data, generates time-based features, handles nulls, ensures Train/Test schema consistency, and saves optimized Parquet files.

```bash
python scripts/run_pipeline.py preprocess
```

#### 🔹 Step 2: Hyperparameter Tuning (Optuna)

Runs Bayesian optimization to find the best parameters for each model. Results are logged to MLflow.

```bash
# Tune all models sequentially
python scripts/run_pipeline.py tune --model all

# Tune a specific model
python scripts/run_pipeline.py tune --model xgboost
```

#### 🔹 Step 3: Training

Retrieves the best parameters from MLflow (or config defaults), trains the final models on the full dataset, and registers them in the MLflow Model Registry with the `@Champion` alias.

```bash
python scripts/run_pipeline.py train --model all
```

#### 🔹 Step 4: Validation (Ensemble)

Loads the registered `@Champion` models, predicts on the validation set, and calculates the Ensemble AUC.

```bash
python scripts/run_pipeline.py ensemble
```

#### 🔹 Step 5: Inference (Submission)

Generates the final `submission.csv` for Kaggle using the weighted ensemble.

```bash
python scripts/run_pipeline.py predict
```

### ⚡ The "One-Click" Command

To run the entire pipeline from start to finish (ETL -> Train -> Validation -> Inference):

```bash
python scripts/run_pipeline.py all
```

## 📊 Performance

* **Single Model (XGBoost Tuned):** AUC ~0.922 (Time-based Validation).
* **Ensemble:** Targeting >0.94 AUC.
* **Hardware:** Optimized for NVIDIA RTX 4080 / AMD Ryzen 9800X3D.

---
*Project developed for the IEEE-CIS Fraud Detection Challenge.*
