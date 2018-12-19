from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Load Data
dataset = pd.read_csv('D:/data/csv/dataset/bu_jelly_mod_nm.csv')
data_columns = list(dataset.columns.values)
X = np.array( dataset[dataset.columns[0:22]])
y = np.array( dataset[dataset.columns[22:]] )

X_valid = X[220:]
X = X[:220]

y_valid = y[220:]
y = y[:220]

accs = []


data = [[i for i in range(100)]]
data = np.array(data, dtype=float)


target = [[i for i in range(1, 101)]]
target = np.array(target, dtype=float)


# data = data.reshape((1, 1, 100))
# target = target.reshape((1, 1, 100))
# print(data)
# print(target)
#
# X_test = [i for i in range(100, 200)]
# X_test = np.array(X_test).reshape((1, 1, 100))
# y_test = [i for i in range(101, 201)]
# y_test = np.array(y_test).reshape((1, 1, 100))
#
# print(X_test)
# print(y_test)



kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(LSTM(54, input_shape=(22, 6), return_sequences=True))
    model.add(Dense(units=6, activation='softmax'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=2)

    predict = model.predict(X_test)

    scores = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy : " + str(scores[1]*100))
    accs.append(scores[1]*100)

sum = 0
for num in accs:
    sum+=num

print("Total Accuracy : " + str( sum/len(accs) ))