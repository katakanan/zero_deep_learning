import numpy as np

def sum_square_error(y, t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid(x):
    return 1/(1+np.exp(-x))
    # return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

def identity_function(x):
    return x

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a-c)
#     sum_exp_a = np.sum(a)
#     y = exp_a / sum_exp_a
#     return y

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x