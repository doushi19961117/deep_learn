# -*- coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import os

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization
from keras.optimizers import adamax, nadam, adam, Adam, SGD
from keras.models import Model
from keras.models import load_model

from PIL import Image
import matplotlib.gridspec as gridspec


class ConvolutionalAutoencoder(object):

    def __init__(self, img_width, img_height, img_channels, img_floder_path, model_path, epochs, batch_size,
                 model_exist=False):
        '''
           用来设置初始化参数
        '''
        # 图片参数设置
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.img_input = (self.img_width, self.img_height, self.img_channels)
        self.img_floder_path = img_floder_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_exist = model_exist

        # 模型不存在则直接训练模型
        if self.model_exist == False:
            self.autoencoder = self.convolutional_autoencoder_model()
            # self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        else:
            self.autoencoder = load_model(self.model_path)
        optimizer = Adam(lr=0.005)  # SGD(lr=0.01, clipnorm=1.)
        self.autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

    # 读取图片称为模型训练的格式
    def __read_img(self):
        # 获取文件夹下所有文件名
        imgs = os.listdir(self.img_floder_path)

        img_width = self.img_input[0]
        img_height = self.img_input[1]
        img_channels = self.img_input[2]

        train_data = []  # 先将shape好的图片先存进train_data，最后在转换为np
        for i in range(len(imgs)):  # len(imgs)
            img = Image.open(os.path.join(self.img_floder_path, imgs[i]))
            img = np.array(img)
            img = np.expand_dims(img, axis=4).reshape(img_width, img_height, img_channels)  # 将img转换成制定形状
            train_data.append(img)
        train_data = np.array(train_data)
        # print(train_data.shape)
        return train_data

    def train(self):

        # 对训练集缩放到[0,1]或者[-1,1]对结果的影响？
        x_train = self.__read_img() / 255

        # 模型训练
        self.autoencoder.fit(x_train, x_train, epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

        # 保存模型
        self.autoencoder.save(self.model_path)
        # 显示样本结果和重构的结果
        # self.predict_result(x_train, self.autoencoder)

    def convolutional_autoencoder_model(self):

        activation_function = 'relu'
        padding_value = 'valid'
        momentum_value = 0.8
        # 搭建ConvolutionalAutoencoder模型
        input_img = Input(shape=(self.img_width, self.img_height, self.img_channels))
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(input_img)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = MaxPooling2D((2, 2))(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = MaxPooling2D((2, 2))(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(8, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)

        encoded = Conv2D(4, (3, 3), activation=activation_function, padding='same', name='encoded')(x)
        # 计算要压缩的特征维度
        img_features = np.asarray(encoded.shape[1:], dtype=np.int)

        # 需要计算压缩的维度
        x = Flatten(name='img_features')(encoded)
        x = Reshape(target_shape=img_features)(x)

        x = Conv2D(8, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(8, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = UpSampling2D((2, 2))(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = UpSampling2D((2, 2))(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        # x = BatchNormalization(momentum=momentum_value)(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)
        x = Conv2D(64, (3, 3), activation=activation_function, padding='same')(x)
        x = BatchNormalization(momentum=momentum_value)(x)

        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        adam = Adam(lr=0.007)
        autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
        autoencoder.summary()
        return autoencoder

    def predict_result(self, x_test, autoencoder):
        decoded_imgs = autoencoder.predict(x_test)

        # 展示的列数
        column = len(x_test)

        fig = plt.figure()
        spec2 = gridspec.GridSpec(ncols=column, nrows=2, hspace=0.1, wspace=0.1)

        for i in range(column):
            # 显示原始图片
            f2_ax1 = fig.add_subplot(spec2[0, i])
            f2_ax1.imshow(x_test[i].reshape(self.img_width, self.img_height, self.img_channels))
            f2_ax1.axis('off')

            # 显示重构图片
            f2_ax2 = fig.add_subplot(spec2[1, i])
            f2_ax2.imshow(decoded_imgs[i].reshape(self.img_width, self.img_height, self.img_channels))
            f2_ax2.axis('off')

        plt.show()


if __name__ == '__main__':
    autoencoder = ConvolutionalAutoencoder(256, 256, 3,
                                           img_floder_path='C:\\Users\\yangqing\\Desktop\\total_img',
                                           model_path='convolutional_autoencoder.h5',
                                           epochs=100,
                                           batch_size=5,
                                           model_exist=False)
    autoencoder.train()
