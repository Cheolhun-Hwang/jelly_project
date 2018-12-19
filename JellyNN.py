import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

dataset = pd.read_csv('D:/data/csv/dataset/bu_jelly_mod_nm.csv')
data_columns = list(dataset.columns.values)
X = np.array( dataset[dataset.columns[0:22]])
y = np.array( dataset[dataset.columns[22:]] )

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 21
    self.outputSize = 6
    self.hiddenSize = 22

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)


accurs = []
accurs2 = []
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(X_train.shape)
    print(y_train.shape)

    model = MLPClassifier(hidden_layer_sizes=(22,), batch_size='auto', activation='relu', solver='lbfgs', alpha=1e-5,
                          random_state=1)
    model2 = MLPClassifier(hidden_layer_sizes=(22,), batch_size='auto', activation='relu', solver='adam', alpha=1e-5,
                           random_state=1)

    model.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    scores = model.score(X_test, y_test)
    scores2 = model.score(X_test, y_test)

    print("model 1 : " +str(scores))
    print("model 2 : " + str(scores2))

    accurs.append(scores*100)
    accurs2.append(scores2*100)


sum=0
for num in accurs:
    sum+=num

print("Model 1 : " + str(sum/len(accurs))+"%")

sum=0
for num in accurs2:
    sum+=num

print("Model 2 : " + str(sum/len(accurs2))+"%")

