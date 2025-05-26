# Telco Customer Churn Prediction using PySpark and MLflow

This project demonstrates how to build an end-to-end churn prediction pipeline using PySpark on Databricks and track experiments using MLflow.

---

##  Project Overview

A telecom company wants to reduce customer churn by identifying which customers are likely to leave. This notebook performs:

- Data loading from Databricks File System (DBFS)
- Feature engineering
- ML pipeline creation
- Model training using Logistic Regression
- Model evaluation using AUC
- Model tracking using MLflow

---

##  Dataset

- **File name:** `WA_Fn_UseC__Telco_Customer_Churn.csv`
- **Location:** `/FileStore/tables/`
- **Source:** [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 🛠️ Key Steps

### 1. **Data Loading**
- Loaded CSV from DBFS using `spark.read.csv()`

### 2. **Data Cleaning**
- Converted `TotalCharges` to `double`
- Handled missing/null values
- Dropped irrelevant columns like `customerID`

### 3. **Feature Engineering**
- Converted `Churn` into binary label
- Indexed and one-hot encoded categorical columns
- Assembled all features using `VectorAssembler`

### 4. **Modeling**
- Used `LogisticRegression` from `pyspark.ml.classification`
- Split into train/test datasets (70/30)
- Evaluated using AUC (Area Under ROC Curve)

### 5. **MLflow Integration**
- Tracked:
  - Model
  - AUC score
  - Parameters (like `regParam`, `maxIter`)
- Logged Spark model via `mlflow.spark.log_model`

---

