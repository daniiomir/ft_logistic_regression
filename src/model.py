import numpy as np

class LogisticRegression:
    def __init__(self, eta):
        self.eta = eta # learning rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self):
        pass

    def predict(self):
        pass
