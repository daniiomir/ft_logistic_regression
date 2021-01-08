import pickle
import numpy as np
import matplotlib.pyplot as plt


class MultiClassLogisticRegression:
    def __init__(self, eta=0.1, n_iter=100, thres=1e-3, random_state=42, verbose=False):
        self.n_iter = n_iter
        self.thres = thres
        self.eta = eta
        self.verbose = verbose
        self.loss = []
        self.classes = None
        self.class_labels = None
        self.weights = None
        np.random.seed(random_state)

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_labels = {c: i for i, c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))
        for epoch in range(self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            update = self.eta * np.dot((y - self.predict_(X)).T, X)
            self.weights += update
            if np.abs(update).max() < self.thres:
                break
            if epoch % 1000 == 0 and self.verbose:
                print(f'Training accuracy at {epoch} iterations is {self.evaluate_(X, y)}')

    def predict(self, X):
        return self.predict_(self.add_bias(X))

    def predict_(self, X):
        return self.softmax(np.dot(X, self.weights.T).reshape(-1, len(self.classes)))

    def predict_classes(self, X):
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.predict(X), axis=1))

    def add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))

    def cross_entropy(self, y, probs):
        return -1 * np.mean(y * np.log(probs))

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)

    def loss_plot(self):
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.title("Development of loss during training")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.savefig('loss_lot.png')

    def save_weights(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f, protocol=4)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)
