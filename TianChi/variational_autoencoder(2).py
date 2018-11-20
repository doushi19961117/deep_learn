# -*- coding:utf-8 -*

'''
   1.用一个类完成对vae的封装，能够对给定的文件夹下的图片进行特征压缩
   2.实现的功能
     2.1 vae整体模型训练
     2.2 输入图片和压缩后生成图片的对比
     2.3 完成对输入图片的压缩来提取特征
     2.4 输入压缩特征来生成图片
    refrence:
'''

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.losses import mae
from keras import backend as K

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import sys
import os


def read_img(folder_path, img_input):
    # 获取文件夹下所有文件名
    imgs = os.listdir(folder_path)

    img_width = img_input[0]
    img_height = img_input[1]
    img_channels = img_input[2]

    train_data = []  # 先将shape好的图片先存进train_data，最后在转换为np
    for i in range(1):
        img = Image.open(os.path.join(folder_path, imgs[i]))
        img = np.array(img)
        # print(img.shape)
        img = np.expand_dims(img, axis=4).reshape(img_width, img_height, img_channels)  # 将img转换成制定形状
        train_data.append(img)
    train_data = np.array(train_data)
    # print(train_data.shape)
    return train_data


class VariationalAutoEncoder(object):

    # 初始化参数
    def __init__(self, img_input, folder_path, laten_dim, model_path, model_exist=False):
        self.img_input = img_input
        self.folder_path = folder_path
        self.laten_dim = laten_dim
        self.model_path = model_path
        self.model_exist = model_exist

        self.vae = self.__vae()
        if model_exist == True:
            self.vae.load_weights(self.model_path)

        adam = Adam(lr=0.001)
        self.vae.compile(optimizer=adam, loss='binary_crossentropy')

        self.vae.summary()

    # 读取图片称为模型训练的格式
    def __read_img(self):
        # 获取文件夹下所有文件名
        imgs = os.listdir(self.folder_path)

        img_width = self.img_input[0]
        img_height = self.img_input[1]
        img_channels = self.img_input[2]

        train_data = []  # 先将shape好的图片先存进train_data，最后在转换为np
        for i in range(len(imgs)):
            img = Image.open(os.path.join(self.folder_path, imgs[i]))
            img = np.array(img)
            img = np.expand_dims(img, axis=4).reshape(img_width, img_height, img_channels)  # 将img转换成制定形状
            train_data.append(img)
        train_data = np.array(train_data)
        # print(train_data.shape)
        return train_data

    # 训练模型(模型不存在，则构建模型)
    def train(self, epochs, batch_size):
        # 训练数据集进行归一化处理[0,1]
        train_data = self.__read_img() / 225
        vae = self.vae
        for i in range(epochs):
            vae.fit(train_data, train_data, batch_size=batch_size)
        vae.save(self.model_path)

    # 创建vae网络结构
    def __vae(self):
        # 创建encoder
        activation_function = 'relu'
        input = Input(shape=self.img_input)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(input)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x_shape = np.asarray(x.shape[1:], dtype=np.int)

        x = Flatten()(x)
        z_mean = Dense(self.laten_dim, activation=activation_function, name='z_mean')(x)
        z_var = Dense(self.laten_dim, activation=activation_function, name='z_var')(x)
        z = Lambda(self.__sampling, output_shape=[self.laten_dim], name='z')([z_mean, z_var])
        encoder = Model(input, [z_mean, z_var, z], name='ecoder')

        encoder.summary()
        # 创建decoder
        latent_input = Input(shape=(self.laten_dim,), name='z_sampling')
        x = Dense(units=np.prod(x_shape), activation=activation_function)(latent_input)
        x = Reshape(target_shape=x_shape)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation=activation_function, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        decoder = Model(latent_input, x, name='decoder')
        decoder.summary()
        # 创建vae
        output = decoder(encoder(input)[2])
        vae = Model(input, output, name='vae')

        # 设置损失函数、优化器
        restructure_loss = mae(input, input)
        kl_loss = K.exp(z_var) - (1 + z_var) + z_mean
        kl_loss = K.sum(kl_loss, axis=-1)
        vae_loss = K.mean(restructure_loss + kl_loss)
        vae.add_loss(vae_loss)

        return vae

    # 使用重参数技巧来实现采样
    def __sampling(self, args):
        z_mean, z_var = args
        batch = K.shape(z_mean)[0]  #
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + z_var * epsilon

    # 输入部分图片，加载模型来对比生成的效果
    def plot_result(self, valid_data, model):
        # 预测图像
        predict_result = model.predict(valid_data)
        axs = plt.subplots(2, len(valid_data))
        for i in range(len(valid_data)):
            # 显示原始图片
            axs[0, i].imshow(valid_data[i])
            # 显示压缩后图片
            axs[1, i].imshow(predict_result[i])
        plt.show()

    # 输入图片来提取特征
    def generate_img(self, img):
        vae = load_model(self.model_path)
        layer_name = 'z'
        encoder = Model(input=vae.input, output=vae.get_layer(layer_name).output)
        return encoder.predict(img)

    # 输入特征来提取图片
    def extract_latent_features(self, features):
        pass


if __name__ == '__main__':
    vae = VariationalAutoEncoder(img_input=(320, 240, 3),
                                 folder_path='C:\\Users\\yangqing\\Desktop\\img',
                                 laten_dim=100,
                                 model_path='vae.h5',
                                 model_exist=True)
    vae.train(epochs=10,
              batch_size=1)
