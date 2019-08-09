from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.createDataFrame([
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "d"),
    (4, "e"),
    (5, "f"),
    (6, "c"),
    (7, "d")
    ], ["id", "category"])

stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)
encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
for e in encoded.select("id", "categoryVec").take(10):
    print e
