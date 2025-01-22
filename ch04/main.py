import sys, os
# sys.path.append(os.pardir)
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
import pickle

def sum_square_error(y, t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

if __name__ == "__main__":
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) #2 is correct answer
    print(cross_entropy_error(y, t))

    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print(cross_entropy_error(y, t))