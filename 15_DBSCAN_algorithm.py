from sklearn import datasets
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# load the data
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data)

# explore the data
print(df.describe())
bp = df.plot(kind='box')
plt.show()

# perform the clustering
dbClustering = DBSCAN(eps=2.5, min_samples=25)
dbClustering.fit(diabetes.data)

print("clustered\n")
from collections import Counter 
print(Counter(dbClustering.labels_))
