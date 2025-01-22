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

if __name__ == "__main__":
    print(numerical_diff(function_1, 5))
    print(numerical_diff(function_1, 10))

    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))
