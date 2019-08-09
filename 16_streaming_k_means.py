from pyspark.mllib.clustering import StreamingKMeans
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

# continuous training
trainigData = ssc.textFileStream("/training/data/dir").map(Vectors.parse)
testData = ssc.textFileStream("/training/data/dir").map(Vectors.parse)
testData = ssc.textFileStream("/testing/data/dir").map(lambda s:LabeledPoint.parse(s))

model = StreamingKMeans()\
        .setK(3)\
        .setDecayFactor(1.0)\
        .setRandomCenters(dim=3, weight=0.0, seed=42)

model.trainOn(trainingData)
prediction = model.predictOnValues(testData)
print(prediction)
