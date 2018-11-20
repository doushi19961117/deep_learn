# -*- coding: utf-8 -*-
import pandas as pd
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical
#%matplotlib inline
dataset = pd.read_csv("C:\\Users\\yangqing\\Desktop\\train.csv")
y_train = dataset['label']
x_train = dataset.drop('label', axis=1)
del dataset
x_train.describe()
x_train['pixel0'].plot()
dropped_columns = []
for column in x_train.columns:
    if x_train[column].max() == x_train[column].min():
        dropped_columns.append(column)
x_train.drop(dropped_columns, axis = 1, inplace = True)
print('Dropped columns:', len(dropped_columns))
print('New shape of training dataset:', x_train.shape)
for column in x_train.columns:
    if x_train[column].isnull().any():
        print('Null value detected in the feature:', column)
min_train = {}
max_train = {}
for column in x_train.columns:
    min_train[column] = x_train[column].min()
    max_train[column] = x_train[column].max()
    x_train[column] = (x_train[column] - x_train[column].min()) / (x_train[column].max() - x_train[column].min()) # saving amplitudes
x_train = x_train.values
y_train.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=False)
y_train = to_categorical(y_train, num_classes = 10)
print(y_train)
FEATURES = 708
layer_1 = 128
LABELS = 10
model_LR = Sequential()
model_LR.add(Dense(layer_1, input_shape = (FEATURES,)))
model_LR.add(Activation('relu'))
model_LR.add(Dense(LABELS))
model_LR.add(Activation('softmax'))
model_LR.summary()
model_LR.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
EPOCHS = 10
BATCH_SIZE = 100
VALIDATION_SIZE = 0.1
training_history = model_LR.fit(x_train,
                                y_train,
                                batch_size = BATCH_SIZE,
                                epochs = EPOCHS,
                                verbose = 1,
                                validation_split = VALIDATION_SIZE)
# model_LR.get_weights()
# model_LR.getget_weights()
# print(model_LR.get_config())
# print(model_LR.get_weights())