import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('D:/data/csv/dataset/bu_jelly_mod_nm.csv')
data_columns = list(dataset.columns.values)
X = dataset[dataset.columns[0:22]]
y = dataset[dataset.columns[22:]]
print(X.shape)
print(y.shape)
X = np.array(X)
y = np.array(y)
knn1 = KNeighborsClassifier(n_neighbors=9)
knn2 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=9, p=2,
           weights='uniform')

np.random.seed(7)
kf = KFold(n_splits=10)

knn1_ac = []
knn2_ac = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(y_train)
    model1 = knn1.fit(X_train, y_train)
    knn1_ac.append(model1.score(X_test, y_test))
    model2 = knn2.fit(X_train, y_train)
    knn2_ac.append(model2.score(X_test, y_test))

sum = 0.0
for item in knn1_ac:
    sum+=item
print("KNN1 : " + str( sum / len(knn1_ac) ))

sum = 0.0
for item in knn2_ac:
    sum+=item
print("KNN2 : " + str( sum / len(knn2_ac) ))


#1~10  까지 가능
# mod + knn1 : 74.38%
# mod + knn2 : 74.38% 83.32 81.23 83.71 83.72 83.32 83.33

# nom + knn1 : 79.28%
# nom + knn2 : 79.28% 79.26 77.26 81.70 80.10