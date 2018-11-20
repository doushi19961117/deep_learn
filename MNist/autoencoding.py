# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
from sklearn import preprocessing

(x_train, y_train), (x_test, y_test) = mnist.load_data()
one_hot = preprocessing.OneHotEncoder(sparse = False)
y_train = one_hot.fit_transform(y_train.reshape(-1,1))
y_test = one_hot.fit_transform(y_test.reshape(-1,1))
x_train = np.expand_dims(x_train, axis=4).astype('float32')/255
x_test = np.expand_dims(x_test, axis=4).astype('float32')/255

print(x_train)