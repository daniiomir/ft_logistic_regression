import tools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


class OneVSAllClassifier:
    def __init__(self, algo):
        self.algo = algo
        self.list_of_algos = []
        self.classes = None
        self.class_count = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.class_count = len(self.classes)
        y = tools.one_hot_encoding(y)
        for i in range(self.class_count):
            al = self.algo()
            al.fit(x, y[i])
            print(f'Accuracy for model {i + 1} - {al.score(x, y)}')
            self.list_of_algos.append(al)

    def predict(self, x: np.ndarray):
        pass


class LogisticRegression:
    def __init__(self, eta: float = 0.1, multiclass: bool = False, n_iter: int = 100,
                 thres: float = 1e-3, random_state: int = 42, verbose: bool = False):
        self.n_iter = n_iter
        self.thres = thres
        self.eta = eta
        self.verbose = verbose
        self.loss = []
        self.multiclass = multiclass
        self.weights = None
        self.classes = None
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

    def score(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) == y)

    def loss_plot(self):
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.title("Development of loss during training")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.savefig('loss_plot.png')

    # tools

    def _fit_multi(self, x, y):  # Y should be in one hot encoding
        self.weights = self._init_weights((self.class_count, x.shape[1]), 'zeros')
        for epoch in range(self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_proba_multi_(x)))
            update = self.eta * np.dot((y - self.predict_proba_multi_(x)).T, x)
            self.weights += update
            if np.abs(update).max() < self.thres:
                break
            if epoch % 10 == 0 and self.verbose:
                print(f'[{epoch}] Loss - {self.loss[-1]}, accuracy - {self.eval(x, y)}')

    def _fit_binary(self, x, y):
        self.weights = self._init_weights((1, x.shape[1]), 'zeros')
        for epoch in range(self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_proba_binary_(x)))
            update = self.eta * np.mean(np.dot(x.T, self.sigmoid(np.dot(self.weights, x)) - y))
            self.weights += update
            if np.abs(update).max() < self.thres:
                break
            if epoch % 10 == 0 and self.verbose:
                print(f'Training accuracy at {epoch} iterations is {self.eval(x, y)}')

    def _init_weights(self, shape: tuple, type: str):
        if type == 'zeros':
            return np.zeros(shape=shape)
        elif type == 'random':
            return np.random.rand(shape[0], shape[1])
        elif type == 'gaussian':
            return np.random.normal(0, 1, shape)
        else:
            raise NotImplementedError

    def add_bias(self, x):
        return np.insert(x, 0, 1, axis=1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def cross_entropy(self, y, probs):  # TODO - implement LOG LOSS
        logloss = log_loss(y, probs)
        # probs[probs == 1] = 0.999
        # if not self.multiclass:
        #     return -y * np.log(probs) - (1 - y) * np.log(1 - probs)
        #     return -1 * np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))  # log loss / binary cross-entropy
        # return -1 * np.mean(y * np.log(probs))  # multiclass cross-entropy
        return logloss

    def predict_proba_binary_(self, x):
        return self.sigmoid(np.dot(self.weights.T, x))

    def predict_proba_multi_(self, x):
        return self.softmax(np.dot(x, self.weights.T).reshape(-1, self.class_count))

    def eval(self, x, y):
        if not self.multiclass:
            proba = self.predict_proba_binary_(x)
        else:
            proba = self.predict_proba_multi_(x)
        return np.mean(np.argmax(proba, axis=1) == np.argmax(y, axis=1))
