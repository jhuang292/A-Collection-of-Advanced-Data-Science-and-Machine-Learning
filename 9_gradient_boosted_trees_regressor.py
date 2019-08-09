# gradient boosted regression
#
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row

from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.mllib.evaluation import RegressionMetrics

sc = SparkContext()
sqlContext = SQLContext(sc)

# load a toy dataset with svm format
df = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()

# Map labels into an indexed column labels [0, numLabels]
stringIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
siModel = stringIndexer.fit(df)
td = siModel.transform(df)

# get the transformed model and split in train and test 
[train, test] = td.randomSplit([0,7, 0.3])

# get the gradient boosted regressor
rf = GBTRegressor(maxIter=50, maxDepth=5, labelCol="indexedLabel")
model = rf.fit(train)

# predict
predictionAndLabels = model.transform(test).select("prediction", "indexedLabel")\
        .map(lambda x: (x.prediction, x.indexedLabel))

# compute metrics
metrics = RegressionMetrics(predictionAndLabels)
print("rmse %.3f" % metrics.rootMeanSquaredError)
print("r2 %.3f" % metrics.r2)
print("mae %.3f" % metrics.meanAbsoluteError)
