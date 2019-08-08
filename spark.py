from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row

from pyspark.ml.feature import HashingTF, Tokenier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# create a toy training set and store in a dataframe 
# (id, text, label) tuples.
sc = SparkContext()
sqlContext = SQLContext(sc)
training = sqlContext.createDataFrame([
    (0L, "the eagles touch base", 1.0),
    (1L, "matt dillon play movies", 0.0),
    (2L, "touch down at 10", 1.0),
    (3L, "tom cruise and", 0.0),
    (4L, "baseball tournament", 1.0),
    (5L, "angeline jolie", 0.0)],
    ["id", "text", "label"])

# ML pipeline, tree stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")

# build the estimator
lr = LogisticRegression(maxIter=10, regParam=0.01)

# a simple pipeline only one stage
pipeline = Pipeline().setStages([tokenizer, hashingTF, lr])

# build the parameter grid
paramGrid = ParamGridBuilder().baseOn({lr.labelCol: 'label'}).baseOn({lr.predictionCol, 'predic'}).addGrid(lr.regParam, [1.0, 2.0]).addGrid(lr.maxIter, [1,5]).build()
expected = [{lr.regParam: 1.0, lr.maxIter:1, lr.labelCol: 'label', lr.predictionCol: 'predic'},
        {lr.regParam: 2.0, lr.maxIter: 1, lr.labelCol: 'label', lr.predictionCol: 'predic'},
        {lr.regParam: 1.0, lr.maxIter: 5, lr.labelCol: 'label', lr.predictionCol: 'predic'},
        {lr.regParam: 2.0, lr.maxIter: 5, lr.labelCol: 'label', lr.predictionCol: 'predic'}]

len(paramGrid) == len(expected)

bce = BinaryClassificationEvaluator()

# the crossvalidator takes the pipeline, the grid, and the evaluator
# run on 2+ folds

cv = CrossValidator().setEstimator(pipeline).setEstimatorParaMap(paramGrid).setEvaluator(bce).setNumFolds(2)

cvModel = cv.fit(training)

print("Parameters lr")
print(lr.extractParamMap())
print("Parameters cvmodel")
print(cv.getEstimatorParamMaps())

# create the toy test documents

test = sqlContext.createDataFrame([(4L, "tom cruise"),])
