from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target
yint = y[:].astype(int)

#grid = 20 x 2 x 3
classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2)
grid = {'n_neighbors': range(1,11), 'weights': ['uniform', 'distance'], 'p': [1,2]}

print('baseline %.5f' % np.mean(cross_val_score(classifier, X, yint, cv=10, scoring='accuracy', n_jobs=1)))

search = GridSearchCV(estimator=classifier, param_grid=grid, scoring='accuracy', n_jobs=1, refit=True, cv=10)
search.fit(X, yint)

print('Best Parameters: %s' % search.best_params_)
print('Accuracy %5f' % search.best_score_)








