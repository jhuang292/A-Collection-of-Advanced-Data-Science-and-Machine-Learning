import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.svm import SVC

diabets = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabets.data, diabets.target, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
clf = SVC(kernel='linear', C=1)
score = cross_val_score(clf, diabets.data, diabets.target, cv=4) # 4-folds
print(score)
print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std()))
