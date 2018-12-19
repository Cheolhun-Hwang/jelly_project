## bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.tree import export_graphviz
from  graphviz import Source
import bitstring
import pandas as pd
from IPython.display import display
import numpy as np
import csv
from sklearn.model_selection import KFold

dataset = pd.read_csv('D:/data/csv/dataset/bu_jelly_mod_nm.csv')
data_columns = list(dataset.columns.values)
X = np.array( dataset[dataset.columns[0:22]])
y = np.array( dataset[dataset.columns[22:]] )

np.random.seed(7)
kf = KFold(n_splits=10)
tree = DecisionTreeClassifier()
tree2 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

tree_ac = []
tree2_ac = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    tree.fit(X_train, y_train)
    tree_ac.append(tree.score(X_test, y_test))

    tree2.fit(X_train, y_train)
    tree2_ac.append(tree2.score(X_test, y_test))

sum = 0
for i in tree_ac:
    sum +=i

print("Tree 1 : " + str(sum / len(tree_ac)))

sum = 0
for i in tree2_ac:
    sum +=i
print("Tree 2 : " + str(sum / len(tree2_ac)))

# nom + 바이너리 트리 : 76.46%
# nom + 엔트로피 바이너리 트리 : 83.31%

# mod + 바이너리 트리 : 76.46%
# mod + 엔트로피 바이너리 트리 : 83.31%






