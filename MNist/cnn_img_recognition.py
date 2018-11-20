# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder(sparse = False)
y_train = one_hot.fit_transform(y_train.reshape(-1,1))
y_test = one_hot.fit_transform(y_test.reshape(-1,1))
x_train = np.expand_dims(x_train, axis=4).astype('float32')/255
x_test = np.expand_dims(x_test, axis=4).astype('float32')/255


input = Input((28,28,1))
x = Conv2D(20,(3,3),padding="same")(input)
x = Conv2D(20,(3,3),padding="same")(x)
x = MaxPooling2D((2, 2),padding="same",)(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = MaxPooling2D((2, 2),padding="same",)(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = MaxPooling2D((2, 2),padding="same",)(x)
x = Flatten()(x)
x = Dense(1500, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(500, activation='relu')(x)
x = Dense(300, activation='relu')(x)
x = Dense(10, activation='softmax')(x)
model = Model(input,x)
model.summary()



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#lr学习率 decay学习速率的衰减系数(每个epoch衰减一次) momentum动量项 Nesterov是否使用Nesterov momentum
model.fit(x_train, y_train, epochs = 10, batch_size = 60) #训练模型1000次，20个一组
score = model.evaluate(x_test, y_test,batch_size=20, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
