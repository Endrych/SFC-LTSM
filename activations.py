import numpy as np


def activation(activation_fcn, value):
    if activation_fcn == "sigmoid":
        return sigmoid(value)
    elif activation_fcn == "tanh":
        return np.tanh(value)
    elif activation_fcn == 'softmax':
        return softmax(value)
    return value


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def softmax(values):
    base = np.exp(values)
    sum_base = np.sum(base)
    return base / sum_base
