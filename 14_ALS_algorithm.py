from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row

from pyspark.ml.recommendation import ALS


sc = SparkContext()
sqlContext = SQLContext(sc)

# create the dataframe (user x item x rating)
df = sqlContext.createDataFrame(
        [(0, 0, 5.0), (0, 1, 1.0), (1, 1, 2.0), (1, 2, 3.0),
            (2, 1, 3.0), (2, 2, 6.0)], 
        ["user", "item", "rating"])

als = ALS(rank=10, maxIter=8)
model = als.fit(df)
print("Rank %i " % model.rank)

test = sqlContext.createDataFrame([(0, 2), (1, 0), (2, 0)],
        ["user", "item"])
predictions = sorted(model.transform(test).collect(), 
        key=lambda r: r[0])
for p in predictions: print p
