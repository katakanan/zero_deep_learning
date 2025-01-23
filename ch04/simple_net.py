import sys, os
# sys.path.append(os.pardir)
sys.path.append(os.getcwd())
import numpy as np
from common.functions import softmax, cross_entropy_error

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
    
