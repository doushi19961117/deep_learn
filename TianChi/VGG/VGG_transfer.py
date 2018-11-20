import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Flatten,MaxoutDense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

train_img_path = 'C:\\Users\\yangqing\\Desktop\\train_img'


def get_VGG_transfer():
    input_img = Input(shape=(224, 224, 3))
    x = Conv2D(100, (3, 3), activation='relu')(input_img)#24
    x = MaxPooling2D()(x)

    x = Conv2D(80, (3, 3), activation='relu')(x)#12
    x = MaxPooling2D()(x)

    x = Conv2D(60, (3, 3), activation='relu')(x)#12
    x = MaxPooling2D()(x)

    x = Conv2D(50, (3, 3), activation='relu')(x)#12
    x = MaxPooling2D()(x)

    x = Conv2D(40, (3, 3), activation='relu')(x)#12
    x = MaxPooling2D()(x)

    x = Conv2D(30, (3, 3), activation='relu')(x)#12
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = MaxoutDense(units=300, activation='relu')(x)
    x = Dense(units=500, activation='relu')(x)
    x = Dense(units=12, activation='sigmoid')(x)

    # 模型编译
    model = Model(inputs=input_img, outputs=x)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



def get_feature():
    base_model = VGG16()

    # VGG模型input=(224,224,3),并提取到block3_pool
    base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    x = base_model.output
    x = Flatten()(x)
    # 模型编译
    model = Model(inputs=base_model.input, outputs=x)

    # 读取img
    img_data = read_img(folder_path=train_img_path, img_input=(224, 224, 3))
    predict = model.predict(img_data, batch_size=5)
    np.savetxt('vgg-teature.txt', predict)


def get_label(folder_path=train_img_path):
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


# 收集数据，之后再进行训练
def train_ae(epochs=500, batch_size=50):
    # 加载模型
    model = get_VGG_transfer()

    # 读取img
    img_data = read_img(folder_path=train_img_path, img_input=(224, 224, 3))

    # 读取标签
    img_label = get_label()
    img_label = np_utils.to_categorical(img_label, num_classes=12)

    # 将数据分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(img_data, img_label, test_size=0.05, random_state=142)

    # 训练模型
    res = []
    for i in range(epochs):
        model.fit(x_train, y_train, batch_size=batch_size)
        loss_and_metrics = model.evaluate(x_test, y_test)
        res.append(loss_and_metrics[1])
    model.save('vgg_transfer.h5')

    plt.plot(range(epochs), res, 'b-')
    plt.show()

# get_feature()
# get_VGG_transfer().summary()
train_ae(epochs=100, batch_size=10)
# data = np.loadtxt('vgg-teature.txt')
