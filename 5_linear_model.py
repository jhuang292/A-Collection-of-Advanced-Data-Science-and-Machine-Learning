from pyspark.ml.feature import RFormula
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)

dataset = sqlContext.createDataFrame(
        [
            (5, "ZA", 1, 1.0),
            (6, "IT", 10, 1.0),
            (7, "US", 18, 1.0),
            (8, "CA", 12, 0.0),
            (9, "NZ", 15, 0.0)],
        ["id", "country", "hour", "clicked"])

formula = RFormula(
        formula="clicked ~ country + hour", 
        featuresCol="features",
        labelCol="label")
output = formula.fit(dataset).transform(dataset)
output.select("features", "label").show()
