from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row
from pyspark import SparkContext 
from pyspark.sql import SQLContext


sc = SparkContext()
sqlContext = SQLContext(sc)

# Load training data
df = MLUtils.loadLibSVMFle(sc,
        "data/mllib/sample_multiclass_classification_data.txt").toDF()

# Map labels into an indexed column labels [0, numLabels]
stringIndexer = StringIndexer(inputCol="label",
        outputCol="indexedLabel")
siModel = stringIndexer.fit(df)
td = siModel.transform(df)

# Split the data into train and test
[train, test] = td.randomSplit([0.6, 0.4], seed = 1234L)

# DecisionTreeClassifier
dt = DecisionTreeClassifier(maxDepth=3, labelCol="indexedLabel")

# train
model = dt.fit(train)

# predict
predictionAndLabels = model.transform(test).select("prediction", "indexedLabel")\
        .map(lambda x: (x.prediction, x.indexedLabel))
