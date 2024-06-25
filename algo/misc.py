import numpy as np


def regularize(value):
    value = value - np.min(value)
    max_v = np.max(value)
    if max_v > 0:
        value /= max_v
    return value


def softmax(x):
    """ softmax function """
    # 为了稳定地计算softmax概率，一般会减掉最大的那个元素
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x
