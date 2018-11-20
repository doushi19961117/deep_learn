import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split

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

train_img_path = 'C:\\Users\\yangqing\\Desktop\\img2\\train_img'
test_img_path = ''
model_path = 'vgg_transfer.h5'
img_shape = (224, 224, 3)


def get_VGG_transfer():
    input_img = Input(shape=img_shape)
    x = Conv2D(100, (3, 3), activation='relu', padding='same')(input_img)  # 24
    x = MaxPooling2D()(x)

    x = Conv2D(80, (3, 3), activation='relu', padding='same')(x)  # 12
    x = MaxPooling2D()(x)

    x = Conv2D(60, (3, 3), activation='relu', padding='same')(x)  # 12
    x = MaxPooling2D()(x)

    x = Conv2D(50, (3, 3), activation='relu', padding='same')(x)  # 12
    x = MaxPooling2D()(x)

    x = Conv2D(40, (3, 3), activation='relu', padding='same')(x)  # 12
    x = MaxPooling2D()(x)

    x = Conv2D(30, (3, 3), activation='relu', padding='same')(x)  # 12
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(units=300, activation='relu')(x)
    x = Dropout(rate=0.02)(x)
    x = Dense(units=500, activation='relu')(x)
    x = Dense(units=12, activation='sigmoid')(x)

    # 模型编译
    model = Model(inputs=input_img, outputs=x)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


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
    img_data = read_img(folder_path=train_img_path, img_input=img_shape)

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
        # 如果在测试集上的效果足够好，就可以提前保存模型
        if loss_and_metrics[1] > 0.88:
            break
        res.append(loss_and_metrics[1])
    model.save(model_path)

    plt.plot(range(epochs), res, 'b-')
    plt.show()


# 预测测试集，并将结果保存在制定文件中
def predict_test_img(model_path=model_path, test_img_path=test_img_path):
    # 加载测试集图片
    test_img = read_img(folder_path=test_img_path, img_input=img_shape)
    # 加载模型
    model = load_model(model_path)
    # 预测数据集
    predict = model.predict_classes(test_img, batch_size=50)
    # 获取测试集图片名称
    test_img_name = os.listdir(test_img_path)
    # 将结果保存到制定文件
    with open('result.csv', 'a', encoding='utf-8') as f:
        for i in range(len(predict)):
            str = test_img_name[i] + ',' + type_transfer.get(predict[i]) + '\n'
            print(str)
            f.write(str)


# predict_test_img(model_path='vgg_transfer.h5', test_img_path=r'C:\Users\10651\Desktop\潘海辉\test_img')

# # 训练模型
train_ae(epochs=5, batch_size=50)
#
# # 预测测试集，并保存结果
# predict_test_img()
