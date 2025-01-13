import os

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("-").getOrCreate()

data_path = "appendicitis.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df.show()

feature_columns = [f"f{i}" for i in range(1, 8)]
df = VectorAssembler(inputCols=feature_columns, outputCol="features").transform(df)

train_data, test_data = df.randomSplit([0.5, 0.5], seed=42)

lr = LogisticRegression(featuresCol="features", labelCol="label")

paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

train_val_split = TrainValidationSplit(
    estimator=lr,
    evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction"),
    estimatorParamMaps=paramGrid_lr,
    trainRatio=0.8
)

lr_model = train_val_split.fit(train_data)
best_lr_model = lr_model.bestModel

predictions_lr = best_lr_model.transform(test_data)
predictions_lr.select("features", "probability", "prediction", "label").show(20, truncate=False)

spark.stop()
