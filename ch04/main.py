import sys, os
# sys.path.append(os.pardir)
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
import pickle
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_diff, numerical_gradient

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


class SimpleNet:
    def __init__(self):
        # self.W = np.random.randn(2, 3)
        self.W = np.array([[0.47355232, 0.9977393, 0.84668094],[0.85557411, 0.03563661, 0.69422093]])
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
    
if __name__ == "__main__":
    net = SimpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0, 0, 1])
    print(net.loss(x, t))

    f = lambda w: net.loss(x, t)
    print(numerical_gradient(f, net.W))
