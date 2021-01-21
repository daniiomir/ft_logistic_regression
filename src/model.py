import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, eta: float = 0.1, multiclass: bool = False, n_iter: int = 100, thres: float = 1e-5, random_state: int = 42, verbose: bool = False):
        self.n_iter = n_iter
        self.thres = thres
        self.eta = eta
        self.verbose = verbose
        self.loss = []
        self.multiclass = multiclass
        self.weights = None
        self.classes = None
        self.class_labels = None
        self.class_count = 0
        np.random.seed(random_state)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.class_count = len(self.classes)
        x = self.add_bias(x)
        if not self.multiclass:
            return self._fit_binary(x, y)
        return self._fit_multi(x, y)

    def predict_proba(self, x: np.ndarray):
        x = self.add_bias(x)
        if not self.multiclass:
            return self.predict_proba_binary_(x)
        return self.predict_proba_multi_(x)

    def predict(self, x: np.ndarray):
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_classes(self, x: np.ndarray):
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.predict(x)))

    def score(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) == y)

    def loss_plot(self):
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.title("Development of loss during training")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.savefig('loss_plot.png')

    # tools

    def _fit_multi(self, x, y):
        self.class_labels = {c: i for i, c in enumerate(self.classes)}
        y = self.one_hot_encoding(y)
        self.weights = self._init_weights((self.class_count, x.shape[1]), 'zeros')
        for epoch in range(self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_proba_multi_(x)))
            update = self.eta * np.dot((y - self.predict_proba_multi_(x)).T, x)
            self.weights += update
            if np.abs(update).max() < self.thres:
                break
            if epoch % 1000 == 0 and self.verbose:
                print(f'Training accuracy at {epoch} iterations is {self.eval(x, y)}')

    def _fit_binary(self, x, y):
        self.weights = self._init_weights((1, x.shape[1]), 'zeros')
        for epoch in range(self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_proba_binary_(x)))
            update = self.eta * np.mean(np.dot(x.T, self.sigmoid(np.dot(self.weights, x)) - y))
            self.weights += update
            if np.abs(update).max() < self.thres:
                break
            if epoch % 1000 == 0 and self.verbose:
                print(f'Training accuracy at {epoch} iterations is {self.eval(x, y)}')

    def _init_weights(self, shape: tuple, type: str):
        if type == 'zeros':
            return np.zeros(shape=shape)
        elif type == 'random':
            return np.random.rand(shape[0], shape[1])
        else:
            raise NotImplementedError

    def add_bias(self, x):
        return np.insert(x, 0, 1, axis=1)

    def one_hot_encoding(self, y):
        return np.eye(self.class_count)[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def cross_entropy(self, y, probs):
        if not self.multiclass:
            return -1 * np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))  # log loss / binary cross-entropy
        return -1 * np.mean(y * np.log(probs))  # multiclass cross-entropy

    def predict_proba_binary_(self, x):
        return self.sigmoid(np.dot(self.weights.T, x))

    def predict_proba_multi_(self, x):
        return self.softmax(np.dot(x, self.weights.T).reshape(-1, self.class_count))

    def eval(self, x, y):
        return np.mean(np.argmax(self.predict_proba(x), axis=1) == np.argmax(y, axis=1))
