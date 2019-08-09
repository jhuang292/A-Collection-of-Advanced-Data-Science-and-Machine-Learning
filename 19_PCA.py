from pyspark.ml.feature import PCA
from pyspark.mllib.linalg import Vectors

from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([5.6, 3.0, 1.0, 6.4, 3.5]),),
        (Vectors.dense([3.4, 5.3, 0.0, 5.5, 6.6]),),
        (Vectors.dense([4.1, 3.1, 3.2, 9.1, 7.0]),),
        (Vectors.dense([3.6, 4.1, 4.2, 6.3, 7.0]),),
        ]

df = sqlContext.createDataFrame(data, ["features"])
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)
result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
