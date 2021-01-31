import src.tools as tools
import numpy as np
import matplotlib.pyplot as plt


class OneVSAllClassifier:
    def __init__(self, algo, **kwargs):
        self.algo = algo
        self.key_args = kwargs
        self.list_of_algos = []
        self.classes = None
        self.class_count = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.class_count = len(self.classes)
        y = tools.one_hot_encoding(y)
        for i in range(self.class_count):
            al = self.algo(**self.key_args)
            selected_y = y[:, i:i + 1][:, 0]
            al.fit(x, selected_y)
            print(f'Accuracy for model {i + 1} - {al.score(x, selected_y)}\n')
            self.list_of_algos.append(al)

    def predict(self, x: np.ndarray):
        preds = []
        for model in self.list_of_algos:
            preds.append(model.predict_proba(x, add_bias=True))
        preds = np.vstack(tuple(preds))
        return np.argmax(preds, axis=0)

    def score(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) == y)

    def loss_plot(self):
        for index, model in enumerate(self.list_of_algos):
            plt.plot(np.arange(len(model.loss)), model.loss, label=f'Model {index + 1}')
        plt.title("Development of loss during training")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.savefig('loss_plot.png')


class LogisticRegression:
    def __init__(self, eta: float = 0.1, multiclass: bool = False, n_iter: int = 100,
                 update_weights_thres: float = 1e-3, random_state: int = 42,
                 verbose: bool = False, verbose_epoch: int = 10, decision_thres: float = 0.5):
        self.n_iter = n_iter
        self.update_weights_thres = update_weights_thres
        self.decision_thres = decision_thres
        self.eta = eta
        self.verbose = verbose
        self.verbose_epoch = verbose_epoch
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
        self._fit(x, y)

    def predict_proba(self, x: np.ndarray, add_bias=False):
        if add_bias:
            x = self.add_bias(x)
        if not self.multiclass:
            return self.predict_proba_binary_(x)
        return self.predict_proba_multi_(x)

    def predict(self, x: np.ndarray):
        if not self.multiclass:
            preds = np.zeros(len(x))
            preds[self.predict_proba(x, add_bias=True) >= self.decision_thres] = 1
            return preds
        return np.argmax(self.predict_proba(x, add_bias=True), axis=1)

    def score(self, x: np.ndarray, y: np.ndarray):
        return np.mean(self.predict(x) == y)

    def loss_plot(self):
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.title("Development of loss during training")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.savefig('loss_plot.png')

    def _fit(self, x, y):
        if not self.multiclass:
            self.weights = np.zeros(x.shape[1])
        else:
            self.weights = np.zeros((self.class_count, x.shape[1]))
            y = tools.one_hot_encoding(y)
        for epoch in range(self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_proba(x, add_bias=False)))
            if not self.multiclass:
                update = self.eta * np.dot(x.T, (y - self.predict_proba_binary_(x)))
            else:
                update = self.eta * np.dot((y - self.predict_proba_multi_(x)).T, x)
            self.weights += update
            if epoch % self.verbose_epoch == 0 and self.verbose:
                print(f'[{epoch}] Loss - {self.loss[-1]}, accuracy - {self.eval(x, y)}')
            if np.abs(update).max() < self.update_weights_thres:
                break

    def add_bias(self, x):
        return np.insert(x, 0, 1, axis=1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def cross_entropy(self, y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        if not self.multiclass:
            return -1 * np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return -1 * np.mean(y_true * np.log(y_pred))

    def predict_proba_binary_(self, x):
        return self.sigmoid(np.dot(x, self.weights.T))

    def predict_proba_multi_(self, x):
        return self.softmax(np.dot(x, self.weights.T).reshape(-1, self.class_count))

    def eval(self, x, y):
        if not self.multiclass:
            preds = np.zeros(len(y))
            preds[self.predict_proba_binary_(x) >= self.decision_thres] = 1
            return np.mean(preds == y)
        else:
            return np.mean(np.argmax(self.predict_proba_multi_(x), axis=1) == np.argmax(y, axis=1))
