import pandas as pd
import numpy as np
from matplotlib.colors import Normalize
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

dataset = pd.read_csv('D:/data/csv/dataset/bu_jelly_mod_nm.csv')
data_columns = list(dataset.columns.values)
X = np.array( dataset[dataset.columns[0:22]])
y = np.array( dataset[dataset.columns[22:]] )

# print(X.head())
# print(y.head())

C = 0.5  # SVM regularization parameter

model_linear_1 = svm.SVC(kernel='linear', C=0.5)
model_linear_2 = svm.LinearSVC(C=0.5)
model_rbf = svm.SVC(kernel='rbf', gamma=10, C=0.5)
model_poly = svm.SVC(kernel='poly', degree=2, C=0.5)

models = (model_linear_1,
          model_linear_2,
          model_rbf,
          model_poly)

# models = (clf.fit(X, y) for clf in models)

l1=[]
l2=[]
rbf=[]
poly=[]

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("## Linear 1'")
    model_linear_1.fit(X_train, y_train)
    l1.append(model_linear_1.score(X_test, y_test))
    print("훈련 세트 정확도: {:.2f}".format(model_linear_1.score(X_train, y_train)))
    print("테스트 세트 정확도: {:.2f}".format(model_linear_1.score(X_test, y_test)))
    print("## Linear 2'")
    model_linear_2.fit(X_train, y_train)
    l2.append(model_linear_2.score(X_test, y_test))
    print("훈련 세트 정확도: {:.2f}".format(model_linear_2.score(X_train, y_train)))
    print("테스트 세트 정확도: {:.2f}".format(model_linear_2.score(X_test, y_test)))
    print("## RBF'")
    model_rbf.fit(X_train, y_train)
    rbf.append(model_rbf.score(X_test, y_test))
    print("훈련 세트 정확도: {:.2f}".format(model_rbf.score(X_train, y_train)))
    print("테스트 세트 정확도: {:.2f}".format(model_rbf.score(X_test, y_test)))
    print("## Poly'")
    model_poly.fit(X_train, y_train)
    poly.append(model_poly.score(X_test, y_test))
    print("훈련 세트 정확도: {:.2f}".format(model_poly.score(X_train, y_train)))
    print("테스트 세트 정확도: {:.2f}".format(model_poly.score(X_test, y_test)))

sum = 0.0
for item in l1:
    sum+=item
print("L1 : " + str( sum / len(l1) ))

sum = 0.0
for item in l2:
    sum+=item
print("L2 : " + str( sum / len(l2) ))

sum = 0.0
for item in rbf:
    sum+=item
print("rbf : " + str( sum / len(rbf) ))

sum = 0.0
for item in poly:
    sum+=item
print("poly : " + str( sum / len(poly) ))

# 각 정확도 1.0
# nom+L1 : 84.57
# nom+L2 : 75.68
# nom+rbf : 84.57
# nom+poly : 80.05

# 각 정확도 : 0.5
# nom+L1 : 84.57
# nom+L2 : 82.07
# nom+rbf : 84.57
# nom+poly : 79.23

# mod+L1 C:1 : 80.07
# mod+L2 C:1 : 81.75
# mod+rbf C:0.5 gamma:10 : 84.57
# mod+poly C:1 degree:2 : 78.05

# mod+L1 C:0.5 : 80.07
# mod+L2 C:0.5 : 80.85
# mod+rbf C:0.5 gamma:10 : 84.57
# mod+poly C:0.5 degree:2 : 78.05



## poly Degree
## poly C:1 degree:1 : 0.8456666666666666
## poly C:1 degree:2 : 0.8456666666666666
## poly C:1 degree:3 : 0.8005000000000001
## poly C:1 degree:4 : 0.7725
## poly C:1 degree:5 : 0.7641666666666667
## poly C:1 degree:6 : 0.76