import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime as dt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

jelly_nm_x = []
jelly_nm_y = []
jelly_nm_column = []
jelly_nm = []

def loadJeelyData() :
    file = open('D:/data/csv/dataset/bu_jelly_nm.csv', 'r', encoding='utf-8')
    rbr = csv.reader(file)
    for line in rbr:
        if line.count('month') > 0:
            jelly_nm_column.extend(line[1:])
        else :
            jelly_nm.append(line)
            jelly_nm_x.append(line[1:19])
            jelly_nm_y.append(line[20])

## Data Load
loadJeelyData()
# for node in jelly_nm_x:
#     print(node)
# for node in jelly_nm_y:
#     print(node)
# print (jelly_nm_column)

X = np.array(jelly_nm_x, dtype=float)
y = np.array(jelly_nm_y, dtype=float)

X_size, y_size = len(X), len(y)

print("data X size : " + str(X_size))
print("data y Size : " + str(y_size))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print("X_train size : " + str(len(X_train)))
print("y_train size : " + str(len(y_train)))
print("X_test size : " + str(len(X_test)))
print("y_test Size : " + str(len(y_test)))

class DataGeneratorSeq(object):
    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data,batch_labels

    def unroll_batches(self):

        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))



dg = DataGeneratorSeq(jelly_nm,18,1)
u_data, u_labels = dg.unroll_batches()

for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ',dat )
    print('\n\tOutput:',lbl)