import numpy as np                  # NumPy
import pandas as pd
from PIL import Image
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn.model_selection import KFold

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(60000,28,28,1)
# X_test = X_test.reshape(10000,28,28,1)
#
# print (X_train)
# print(X_train[0].shape)
#

# Image resize
resize_w = 16
resize_h = 16
# nom : normalize, mod : original
files = 'mod'
# kernel-size
ks = 3
# batch-size  before : 10
bs = 16
# epochs  before : 10
ep = 20

def cnn_model():
    # create model
    model = Sequential()

    model.add(Convolution2D(32, kernel_size=(ks, ks), strides=(1, 1), activation='relu', input_shape=(1, resize_w, resize_h)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Convolution2D(64, (ks, ks), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(units=6, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



np.random.seed(7)
dataset = pd.read_csv('D:/data/csv/dataset/bu_jelly_'+files+'_nm_byte.csv')
data_columns = list(dataset.columns.values)
X = dataset[dataset.columns[0]]
y = dataset[dataset.columns[1]]

X = np.array(X)
y_s = np.array(y)
imgs = []
path = "D:/data/csv/dataset/nm_"+files+"_image/25x70"
valid_images = [".bmp"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    # imgs.append(Image.open(os.path.join(path,f)))
    img = Image.open(os.path.join(path,f))
    img = img.resize((resize_w, resize_h))

    imgs.append(np.array(img, dtype=float))

# print(len(imgs))
# imgs[1].show()





accs = []
losses = []

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))

    x_train_i = []
    x_test_i = []

    for n in train_index :
        x_train_i.append(imgs[n])
    for n in test_index :
        x_test_i.append(imgs[n])

    X_train, X_test = np.array(x_train_i), np.array(x_test_i)
    y_train, y_test = y[train_index], y[test_index]

    X_train = X_train.reshape(X_train.shape[0], 1, resize_w, resize_h).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, resize_w, resize_h).astype('float32')

    y_train = np_utils.to_categorical(y_train, 6)
    y_test = np_utils.to_categorical(y_test, 6)
    num_classes = y_test.shape[1]

    model = cnn_model()
    model.fit(X_train, y_train, epochs=ep, batch_size=bs)
    scores = model.evaluate(X_test, y_test, verbose=0)

    print("Score : %.2f%%" % (scores[1] * 100))
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    accs.append((scores[1] * 100))
    losses.append(scores[0]*100)

sum = 0
sum2 = 0
for num in accs:
    sum += num
for num in losses:
    sum2 += num
print("Kfold Acc :  %.2f%%" % (sum/len(accs)))
print("Kfold Loss :  %.2f%%" % (sum2/len(losses)))
#Test
# X_train = np.array(imgs[0:149])
# X_test = np.array(imgs[149:])
# y_train = y_s[0:149]
# y_test = y_s[149:]
#
# X_train = X_train.reshape(X_train.shape[0], 1, 25, 70).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 1, 25, 70).astype('float32')

# print(X_train.shape[0])

# print (X_train)

# y_train = np_utils.to_categorical(y_train, 6)
# y_test = np_utils.to_categorical(y_test, 6)
# num_classes = y_test.shape[1]


# model = cnn_model()

# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# model.fit(X_train, y_train, epochs=10, batch_size=10)
# scores = model.evaluate(X_test, y_test, verbose=0)
#
# print("Score : %.2f%%" % (scores[1]*100))
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))

## 25x70 mod : 84.56%
## 25x70 nom : 84.56%

## 128x128 mod : %
## 128x128 nom : %

