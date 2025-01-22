import numpy as np

def sum_square_error(y, t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def sigmoid(x):
    return 1/(1+np.exp(-x))
    # return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

def identity_function(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(a)
    y = exp_a / sum_exp_a