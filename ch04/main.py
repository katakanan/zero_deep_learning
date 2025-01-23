import sys, os
# sys.path.append(os.pardir)
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
import pickle
from common.functions import softmax, cross_entropy_error, sigmoid
from common.gradient import numerical_diff, numerical_gradient
from two_layer_net import TwoLayerNet

if __name__ == "__main__":
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)