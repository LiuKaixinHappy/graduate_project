```python
import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D

# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation


np.random.seed(2016)
use_cache = 1
# color type: 1 - grey, 3 - rgb
color_type_global = 3

# color_type = 1 - gray
# color_type = 3 - RGB

# 获取图片（路径，图片行，图片列，图片颜色通道）224, 224
def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    # 如果是灰度图，则用opencv函数读成灰度图，否则彩色图
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)

    # Reduce size
    # 缩减图片大小
    resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    # resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized

# 获得司机信息（标签）
def get_driver_data():
    # 创建字典
    dr = dict()
    # 读取driver_imgs_list.csv
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    # 按行读取，直到读完
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        # 数据格式为{'img_44733.jpg':p002}
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

# 加载训练集
def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []
    # 获取司机数据
    driver_data = get_driver_data()

    # 读取训练集
    print('Read train images')
    # 从c0文件夹读到c9文件夹
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('..', 'input', 'imgs', 'train',
                            'c' + str(j), '*.jpg')
        # 获取path下的所有jpg文件
        files = glob.glob(path)
        for fl in files:
            # 返回path最后的文件名，即xxxx.jpg
            flbase = os.path.basename(fl)
            # 缩减图片大小
            img = get_im(fl, img_rows, img_cols, color_type)
            # 将图片加入训练集
            X_train.append(img)
            # 将类别加入标签y
            y_train.append(j)
            # 将司机信息放入driver_id中
            driver_id.append(driver_data[flbase])

    # driver_id去重排序后放入unique_drivers
    unique_drivers = sorted(list(set(driver_id)))
    # 打印driver数目
    print('Unique drivers: {}'.format(len(unique_drivers)))
    # 打印driver
    print(unique_drivers)
    # 将训练集、标签、训练集图片对应的driver_id、所有driver_id
    return X_train, y_train, driver_id, unique_drivers

# 加载测试集
def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    # 读取测试数据
    path = os.path.join('..', 'input', 'imgs', 'test', '*.jpg')
    # 找到path下的所有jpg文件
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0

    # 所有jpg文件总数除以10向下取整
    thr = math.floor(len(files)/10)
    for fl in files:
        # 获取图片名，xxxx.jpg
        flbase = os.path.basename(fl)
        # 缩减图片大小
        img = get_im(fl, img_rows, img_cols, color_type)
        # 将图片加入测试集
        X_test.append(img)
        # 将图片对应的名称加入X_test_id
        X_test_id.append(flbase)
        # 累计数加1
        total += 1
        # 如果读到批数，输出从xxxx个文件中读取xxxx张图
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    # 返回测试集与测试集图片对应的名称
    return X_test, X_test_id

# 缓存图片（数据，路径）
def cache_data(data, path):
    # 如果cache目录不存在
    if not os.path.isdir('cache'):
        # 创建cache目录
        os.mkdir('cache')
    # 如果路径存在
    if os.path.isdir(os.path.dirname(path)):
        # 打开文件
        file = open(path, 'wb')
        # 将data持久保存到file中
        pickle.dump(data, file)
        # 关闭文件流
        file.close()
    else:
        print('Directory doesnt exists')

# 从缓存读取数据
def restore_data(path):
    # 创建data字典
    data = dict()
    # 如果文件存在
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        # 从文件中读取数据
        data = pickle.load(file)
    return data

# 保存模型
def save_model(model, index, cross=''):
    # 将模型保存为json格式
    json_string = model.to_json()
    # 如果cache目录不存在，创建cache
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    # 模型名
    json_name = 'architecture' + str(index) + cross + '.json'
    # 模型权重名
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)

# 读取模型
def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model

# 拆分验证集（训练集，标签，验证集大小）
def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = \
        train_test_split(train, target,
                         test_size=test_size,
                         random_state=random_state)
    return X_train, X_test, y_train, y_test

# 创建提交文本
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

# 读取并归一化混洗训练数据
def read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                              color_type=1):

    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        train_data, train_target, driver_id, unique_drivers = \
            load_train(img_rows, img_cols, color_type)
        cache_data((train_data, train_target, driver_id, unique_drivers),
                   cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = \
            restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    if color_type == 1:
        train_data = train_data.reshape(train_data.shape[0], color_type,
                                        img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
    # train_data /= 255
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows=224, img_cols=224, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type,
                                      img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]
    # test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('../input/vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model

# 运行交叉验证
def run_cross_validation(nfolds=10, nb_epoch=10, split=0.2, modelStr=''):

    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = 224, 224
    batch_size = 64
    random_state = 20

    train_data, train_target, driver_id, unique_drivers = \
        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
                                                  color_type_global)

    # ishuf_train_data = []
    # shuf_train_target = []
    # index_shuf = range(len(train_target))
    # shuffle(index_shuf)
    # for i in index_shuf:
    #     shuf_train_data.append(train_data[i])
    #     shuf_train_target.append(train_target[i])

    # yfull_train = dict()
    # yfull_test = []
    num_fold = 0
    kf = KFold(len(unique_drivers), n_folds=nfolds,
               shuffle=True, random_state=random_state)
    for train_drivers, test_drivers in kf:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        # print('Split train: ', len(X_train), len(Y_train))
        # print('Split valid: ', len(X_valid), len(Y_valid))
        # print('Train drivers: ', unique_list_train)
        # print('Test drivers: ', unique_list_valid)
        # model = create_model_v1(img_rows, img_cols, color_type_global)
        # model = vgg_bn_model(img_rows, img_cols, color_type_global)
        model = vgg_std16_model(img_rows, img_cols, color_type_global)

        model.fit(train_data, train_target, batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1,
                  validation_split=split, shuffle=True)

        # print('losses: ' + hist.history.losses[-1])

        # print('Score log_loss: ', score[0])

        save_model(model, num_fold, modelStr)

        # predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
        # score = log_loss(Y_valid, predictions_valid)
        # print('Score log_loss: ', score)
        # Store valid predictions
        # for i in range(len(test_index)):
        #    yfull_train[test_index[i]] = predictions_valid[i]

    print('Start testing............')
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
                                                      color_type_global)
    yfull_test = []

    for index in range(1, num_fold + 1):
        # 1,2,3,4,5
        # Store test predictions
        model = read_model(index, modelStr)
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        yfull_test.append(test_prediction)

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(nfolds) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)


def test_model_and_submit(start=1, end=1, modelStr=''):
    img_rows, img_cols = 224, 224
    # batch_size = 64
    # random_state = 51
    nb_epoch = 15

    print('Start testing............')
    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols,
                                                      color_type_global)
    yfull_test = []

    for index in range(start, end + 1):
        # Store test predictions
        model = read_model(index, modelStr)
        test_prediction = model.predict(test_data, batch_size=128, verbose=1)
        yfull_test.append(test_prediction)

    info_string = 'loss_' + modelStr \
                  + '_r_' + str(img_rows) \
                  + '_c_' + str(img_cols) \
                  + '_folds_' + str(end - start + 1) \
                  + '_ep_' + str(nb_epoch)

    test_res = merge_several_folds_mean(yfull_test, end - start + 1)
    create_submission(test_res, test_id, info_string)

# nfolds, nb_epoch, split
run_cross_validation(2, 20, 0.15, '_vgg_16_2x20')

# nb_epoch, split
# run_one_fold_cross_validation(10, 0.1)

# test_model_and_submit(1, 10, 'high_epoch')
```
