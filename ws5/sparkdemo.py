#A1
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("ws5-regression").getOrCreate()

#A2 | cmnd line args
import sys
bucketPath = sys.argv[1]
df = spark.read.csv(bucketPath, header=True, inferSchema=True)
df.show()

#A3
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols=["total_bill","size"], outputCol="features")

#A4
trainDF, testDF = df.randomSplit([.8,.2], seed=42)

#A5
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
lr = LinearRegression(featuresCol="features", labelCol="tip")
pipeline = Pipeline(stages=[vecAssembler,lr])
pipelineModel = pipeline.fit(trainDF)

#A6
predDF = pipelineModel.transform(testDF)
predDF.select("total_bill", "size", "tip", "prediction").show(5)

#A7
from pyspark.ml.evaluation import RegressionEvaluator
regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="tip")
rmse = regressionEvaluator.evaluate(predDF)

regressionEvaluator.setMetricName("r2")
r2= regressionEvaluator.evaluate(predDF) 

#A8
lrModel = pipelineModel.stages[-1]
print(f"Coefficients: {lrModel.coefficients}")
print(f"Intercept: {lrModel.intercept}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
