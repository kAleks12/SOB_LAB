import os

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("-").getOrCreate()

data_path = "appendicitis.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df.show()

feature_columns = [f"f{i}" for i in range(1, 8)]
df = VectorAssembler(inputCols=feature_columns, outputCol="features").transform(df)

lr = LogisticRegression(featuresCol="features", labelCol="label")
paramGrid_lr = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.1, 0.01, 0.001])
    .addGrid(lr.maxIter, [100, 200, 300])
    .build()
)

cross_validator = CrossValidator(
    estimator=lr,
    evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction"),
    estimatorParamMaps=paramGrid_lr,
    numFolds=5
)

lr_model = cross_validator.fit(df)
lr_metrics = lr_model.avgMetrics

print(f"Area under roc for logreg best model -> index: {lr_metrics.index(max(lr_metrics))}; val: {max(lr_metrics)}")

dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
paramGrid_dt = (
    ParamGridBuilder()
    .addGrid(dt.maxDepth, [5, 10, 15])
    .addGrid(dt.maxBins, [5, 10, 15])
    .build()
)

cross_validator = CrossValidator(
    estimator=dt,
    evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction"),
    estimatorParamMaps=paramGrid_dt,
    numFolds=5
)

dt_model = cross_validator.fit(df)
dt_metrics = dt_model.avgMetrics

print(
    f"Area under roc for decision tree best model -> index: {dt_metrics.index(max(dt_metrics))}; val: {max(dt_metrics)}")

spark.stop()
