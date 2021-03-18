import pathlib

import scipy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def download():
    global path
    moving_mnist_url = 'http://www.cs.toronto.edu/%7Enitish/unsupervised_video/mnist_test_seq.npy'
    data_dir = pathlib.Path('data/moving_mnist.npy')
    if not data_dir.exists():
        path = tf.keras.utils.get_file(
            'mnist_test_seq.npy',
            origin=moving_mnist_url,
            # extract=True,
            cache_dir='.', cache_subdir='data')
    return path


def load_data(file_path):
    data = np.load(file_path)
    print(data.shape)  # (20, 10000, 64, 64)
    return data


def split(data, train_cnt, test_cnt):
    # plt.imshow(data[0,0])
    # plt.show()
    data = np.transpose(data, [1, 0, 2, 3])
    data = np.expand_dims(data, 4)
    # print(data.shape)
    # plt.imshow(data[0, 0])
    # plt.show()
    train_set_x = data[:train_cnt, :10, :, :]
    train_set_y = data[:train_cnt, 10:, :, :]
    test_set_x = data[train_cnt:test_cnt, :10, :, :]
    test_set_y = data[train_cnt:test_cnt, 10:, :, :]
    # print(train_set_x.shape)
    # print(train_set_y.shape)
    # print(test_set_x.shape)
    # print(test_set_y.shape)
    return train_set_x, train_set_y, test_set_x, test_set_y


# 将图片矩阵转化为图片并保存到本地
# shape = [batch_size, 10, 64, 64]
def save_as_image(matrix, mode=0):
    if mode == 1:
        root = 'result/prediction'
    elif mode == 0:
        root = 'result/standard'
    else:
        root = 'result/sa_prediction'
    import shutil
    if os.path.exists(root):
        shutil.rmtree(root)
    for testcase_id in range(matrix.shape[0]):
        directory = pathlib.Path(root + '/testcase{}'.format(testcase_id))
        if not directory.exists():
            os.makedirs(directory, exist_ok=True)
            print(os.getcwd())
        testcase = matrix[testcase_id]
        for i in range(matrix.shape[1]):
            img = testcase[i]
            s = ''.join([directory.__str__(), '/{}.jpg'.format(i)])
            print(s)
            plt.imsave(s, img)
