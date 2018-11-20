# -*- coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda
from keras.optimizers import adamax, nadam, adam, Adam, SGD
from keras.models import Model
from keras.models import load_model
from sklearn.manifold import TSNE
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split
from keras.losses import mae, binary_crossentropy

from PIL import Image


def read_img(folder_path, img_input):
    # 获取文件夹下所有文件名
    imgs = os.listdir(folder_path)

    img_width = img_input[0]
    img_height = img_input[1]
    img_channels = img_input[2]

    train_data = []  # 先将shape好的图片先存进train_data，最后在转换为np
    for i in range(len(imgs)):
        img = Image.open(os.path.join(folder_path, imgs[i]))
        img = np.array(img)
        # print(img.shape)
        img = np.expand_dims(img, axis=4).reshape(img_width, img_height, img_channels)  # 将img转换成制定形状
        train_data.append(img)
    train_data = np.array(train_data)
    # print(train_data.shape)
    return train_data


# 获取cae中的encoder来对img进行特征压缩
def get_encoder(model_path):
    layer_name = 'img_features'
    model = load_model(model_path)

    output_layer = output = model.get_layer(layer_name).output
    intermediate_layer_model = Model(input=model.input, output=output_layer)

    return intermediate_layer_model


# 采用T-NSE进行降维可视化
def T_SNE_visualization(model_path, train_img_path):
    # 加载encode
    encoder = get_encoder(model_path)
    # 读取img
    img_data = read_img(folder_path=train_img_path, img_input=(320, 256, 3))

    # 图片预测
    features = encoder.predict(img_data)

    label = get_label(folder_path=train_img_path)
    # 图片降维，可视化
    X_tsne = TSNE(learning_rate=900).fit_transform(features)
    plt.figure(figsize=(12, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, s=8)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.colorbar()
    plt.savefig('res_vae.png')


def get_label(folder_path='C:\\Users\\yangqing\\Desktop\\train_img'):
    # 获取文件夹下所有文件名
    imgs = os.listdir(folder_path)
    labels = []
    for i in range(len(imgs)):

        if '擦花' in imgs[i]:
            label = 2
        elif '漏底' in imgs[i]:
            label = 5
        elif '脏点' in imgs[i]:
            label = 10
        elif '凸粉' in imgs[i]:
            label = 8
        elif '不导电' in imgs[i]:
            label = 1
        elif '横条压凹' in imgs[i]:
            label = 3
        elif '桔皮' in imgs[i]:
            label = 4
        elif '碰伤' in imgs[i]:
            label = 6
        elif '起坑' in imgs[i]:
            label = 7
        elif '涂层开裂' in imgs[i]:
            label = 9
        elif '正常' in imgs[i]:
            label = 0
        else:
            label = 11
        labels.append(label)
    return np.asarray(labels, dtype=np.int)


# 创建ANN来对压缩后的特征进行压缩
def ann(input_dim, output_dim):
    activation_function = 'relu'
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model = Sequential()
    model.add(Dense(units=60, input_dim=input_dim, activation=activation_function))
    model.add(Dense(units=60, activation=activation_function))
    model.add(Dense(units=60, activation=activation_function))
    model.add(Dense(units=60, activation=activation_function))
    model.add(Dense(units=60, activation=activation_function))
    model.add(Dense(units=output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# 从文件名转到标签
type_2_num = {'正常': 0,
              '不导电': 1,
              '擦花': 2,
              '横条压凹': 3,
              '桔皮': 4,
              '漏底': 5,
              '碰伤': 6,
              '起坑': 7,
              '凸粉': 8,
              '涂层开裂': 9,
              '脏点': 10,
              '其他': 11}

# 用来最后提交结果时候的转换
type_transfer = {0: 'norm',
                 1: 'defect1',
                 2: 'defect2',
                 3: 'defect3',
                 4: 'defect4',
                 5: 'defect5',
                 6: 'defect6',
                 7: 'defect7',
                 8: 'defect8',
                 9: 'defect9',
                 10: 'defect10',
                 11: 'defect11'}


# 收集数据，之后再进行训练
def train_ae(model_path = '', train_img_path='', epochs=500, batch_size=50):
    # 加载模型
    ae = get_encoder(model_path)##get_encoder(model_path):

    # 读取img
    img_data = read_img(folder_path=train_img_path, img_input=(320, 256, 3))
    img_data = ae.predict(x = img_data, batch_size=50)

    # 读取标签
    img_label = get_label(train_img_path)
    img_label = np_utils.to_categorical(img_label, num_classes=12)

    print(img_data.shape, img_label.shape)
    # 将数据分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(img_data, img_label, test_size=0.02, random_state=142)

    # 训练模型
    model = ann(input_dim=80, output_dim=12)

    res = []
    for i in range(epochs):
        model.fit(x_train, y_train, batch_size=batch_size)
        loss_and_metrics = model.evaluate(x_test, y_test)
        res.append(loss_and_metrics[1])
    model.save('ann.h5')

    plt.plot(range(epochs), res, 'b-')
    plt.show()
    # print(res)


# --------------------
# training
# --------------------


#进行可是化显示
# T_SNE_visualization(model_path='convolutional_autoencoder.h5',
#                     train_img_path='C:\\Users\\yangqing\\Desktop\\新建文件夹\\train_img')

#训练模型，并显示结果
train_ae(model_path='convolutional_autoencoder.h5',
         train_img_path='C:\\Users\\yangqing\\Desktop\\新建文件夹\\train_img',
         epochs=100,
         batch_size=30)