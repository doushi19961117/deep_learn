# -*- coding: utf-8 -*-
import numpy as np
from keras import Input, Model
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from keras.optimizers import Adam
from sklearn import preprocessing

import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()
one_hot = preprocessing.OneHotEncoder(sparse = False)
y_train = one_hot.fit_transform(y_train.reshape(-1,1))
y_test = one_hot.fit_transform(y_test.reshape(-1,1))
x_train = np.expand_dims(x_train, axis=4).astype('float32')/255
x_test = np.expand_dims(x_test, axis=4).astype('float32')/255

img = x_train[2].reshape(28,28)


input = Input((28,28,1))
x = Conv2D(20,(3,3),padding="same")(input)
x = Conv2D(20,(3,3),padding="same")(x)
x = MaxPooling2D((2, 2),padding="same",)(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = MaxPooling2D((2, 2),padding="same",)(x)
x = Conv2D(40,(3,3),padding="same")(x)
x_shape = np.asarray(x.shape[1:], dtype=np.int)
x = Flatten()(x)
# x = Dense(units=100, activation="relu")(x)
n = np.prod(x_shape)
x = Dense(units=n, activation="relu")(x)
x = Reshape(target_shape=x_shape)(x)
# x = Reshape(target_shape=x_shape)(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = Conv2D(40,(3,3),padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(20,(3,3),padding="same")(x)
x = Conv2D(1,(3,3),padding="same")(x)
model = Model(input,x, name='ecoder')
model.summary()
print(x_train.shape)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, x_train, epochs =1, batch_size = 100) #训练模型1000次，20个一组
predict_result = model.predict(x_test)
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )
ax = ax.flatten()
for i in range(10):
    ax[i].imshow(predict_result[i].reshape(28,28), cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()