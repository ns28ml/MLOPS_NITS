# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

'''# File location and type
file_location = "/FileStore/tables/WA_Fn_UseC__Telco_Customer_Churn.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "yes"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)
'''

# COMMAND ----------

# MAGIC %md
# MAGIC A telecom company wants to reduce customer churn — that is, the number of customers leaving their services. To do that, they want to predict which customers are at risk of churning, based on their account information, usage patterns, and demographics.

# COMMAND ----------

# MAGIC %md
# MAGIC Build a machine learning pipeline to predict churn (Yes/No) using historical customer data.
# MAGIC

# COMMAND ----------

df = spark.read.csv("/FileStore/tables/WA_Fn_UseC__Telco_Customer_Churn.csv", header=True, inferSchema=True)
df.display()


# COMMAND ----------

'''# Create a view or table

temp_table_name = "WA_Fn_UseC__Telco_Customer_Churn_csv"

df.createOrReplaceTempView(temp_table_name)
'''

# COMMAND ----------

'''%sql

/* Query the created temp table in a SQL cell */

select * from `WA_Fn_UseC__Telco_Customer_Churn_csv`
'''

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

#permanent_table_name = "WA_Fn_UseC__Telco_Customer_Churn_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)


# COMMAND ----------

df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering 
# MAGIC
# MAGIC - Fix data types (e.g., convert TotalCharges to double)
# MAGIC - Handle missing/null values
# MAGIC - Drop unnecessary columns (e.g., customerID)

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count


# COMMAND ----------

df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
#df.show()
df.display()

# COMMAND ----------

#check nulls
df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering 
# MAGIC - Convert Churn to binary label (Yes → 1, No → 0)
# MAGIC - Index categorical variables (e.g., Contract, InternetService)
# MAGIC - One-hot encode indexed categorical variables
# MAGIC - Assemble final features into a single vector column

# COMMAND ----------

from pyspark.sql.functions import when, col

df = df.withColumn("Churn_label", when(col("Churn") == "Yes", 1).otherwise(0))
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # StringIndexer, OneHotEncoder

# COMMAND ----------

cat_cols = [col_name for col_name in df.columns if df.select(col_name).dtypes[0][1] == "string" and col_name not in ["Churn", "customerID"]]


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in cat_cols]
encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in cat_cols]


# COMMAND ----------

# MAGIC %md
# MAGIC #Assemble all features

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
feature_cols = [col + "_vec" for col in cat_cols] + numeric_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")


# COMMAND ----------

from pyspark.ml import Pipeline

stages = indexers + encoders + [assembler]
pipeline = Pipeline(stages=stages)

model_df = pipeline.fit(df).transform(df)
df.display()

# COMMAND ----------

model_df = model_df.drop("customerID")


# COMMAND ----------

# MAGIC %md
# MAGIC #Train the model

# COMMAND ----------

print(df.columns)


# COMMAND ----------

feature_columns = [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn', 'Churn_label']


# COMMAND ----------

from pyspark.sql import functions as F


# COMMAND ----------

model_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in feature_columns]).show()


# COMMAND ----------

model_df = df.na.drop(subset=feature_columns)
model_df.display()

# COMMAND ----------

model_df = model_df.na.fill(0, subset=feature_columns)
model_df.display()

# COMMAND ----------

categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod","Churn"
]


# COMMAND ----------

model_df.printSchema()


# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(
    inputCols=[f"{col}_index" for col in categorical_columns],
    outputCols=[f"{col}_vec" for col in categorical_columns]
)


# COMMAND ----------

from pyspark.ml.feature import StringIndexer

indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
    for col in categorical_columns
]


# COMMAND ----------

numeric_columns = [col for col in feature_columns if col not in categorical_columns]


# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[f"{col}_vec" for col in categorical_columns] + numeric_columns,
    outputCol="features"
)

pipeline = Pipeline(stages=indexers + [encoder, assembler])
pipeline_model = pipeline.fit(model_df)
model_df = pipeline_model.transform(model_df)


# COMMAND ----------

# MAGIC %md
# MAGIC #Evaluate the model

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
model_df = label_indexer.fit(model_df).transform(model_df)


# COMMAND ----------

train_df, test_df = model_df.randomSplit([0.7, 0.3], seed=42)


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)


# COMMAND ----------

predictions = lr_model.transform(test_df)
predictions.select("features", "label", "prediction", "probability").show(5)


# COMMAND ----------

predictions.show(5)


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")


# COMMAND ----------

#%pip install --upgrade typing_extensions


# COMMAND ----------

'''import typing_extensions
print(typing_extensions.__version__)
'''

# COMMAND ----------

#pip install --upgrade typing_extensions


# COMMAND ----------

import mlflow
import mlflow.spark



# COMMAND ----------

with mlflow.start_run():
    # Log the model
    mlflow.spark.log_model(lr_model, "model")
    
    # Optionally log metrics or parameters
    mlflow.log_metric("AUC", 0.9999993)


# COMMAND ----------

'''import mlflow
import mlflow.spark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

def load_data(spark):
    # Load data (replace with your data source)
    df = spark.read.csv("/FileStore/tables/WA_Fn_UseC__Telco_Customer_Churn.csv", header=True, inferSchema=True)

#    df = spark.read.format("csv").option("header", "true").load("path/to/data.csv")
    # Preprocess data here (cast columns, handle missing values, etc.)
    # For example:
    # df = df.withColumn("label", df["target"].cast("double"))
    return df

def preprocess_data(df):
    # Implement feature engineering or transformations here
    # For example, assemble features into a vector using VectorAssembler
    from pyspark.ml.feature import VectorAssembler

    feature_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod","Churn"]  # replace with your features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)
    return df

def train_model(df_train):
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.01)
    model = lr.fit(df_train)
    return model, lr.getOrDefault("maxIter"), lr.getOrDefault("regParam")

def evaluate_model(model, df_test):
    predictions = model.transform(df_test)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    return auc

def run_pipeline():
    spark = SparkSession.builder.appName("MLflowPipeline").getOrCreate()

    # Load and preprocess
    data = load_data(spark)
    data = preprocess_data(data)

    # Split data
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    # Start MLflow run
    with mlflow.start_run():
        # Train
        model, max_iter, reg_param = train_model(train)

        # Evaluate
        auc = evaluate_model(model, test)

        # Log parameters, metrics, model
        mlflow.log_param("maxIter", max_iter)
        mlflow.log_param("regParam", reg_param)
        mlflow.log_metric("AUC", auc)
        mlflow.spark.log_model(model, "model")

        print(f"Logged run with AUC: {auc}")

    spark.stop()

if __name__ == "__main__":
    run_pipeline()
'''

# COMMAND ----------

# MAGIC %md
# MAGIC # Practice

# COMMAND ----------

'''from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("EncodingExample").getOrCreate()

data = [
    ("Male", 5),
    ("Female", 10),
    ("Transgender", 3),
    ("Female", 8),
    ("Male", 6)
]

df = spark.createDataFrame(data, ["gender", "tenure"])
df.show()
'''


# COMMAND ----------

# MAGIC %md
# MAGIC # Windows Function

# COMMAND ----------

for column in df.columns:
    print(column)


# COMMAND ----------

df.select("Churn").distinct().show()

# COMMAND ----------

df = df.withColumn(
    "Churn_Int",
    F.when(F.col("Churn") == "Yes", 1).when(F.col("Churn") == "No", 0).otherwise(None)
)


# COMMAND ----------

from pyspark.sql.window import Window

window_spec = Window.partitionBy("tenure")

df_with_churn_rate = df.withColumn(
    "churn_rate_by_tenure",
    F.avg("Churn_Int").over(window_spec)
)

df_with_churn_rate.select("tenure", "Churn", "Churn_Int", "churn_rate_by_tenure").show(100)
