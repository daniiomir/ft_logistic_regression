import numpy as np

class AccuracyScore:
    def __call__(self, y_true, y_pred):
        if isinstance(y_real, np.ndarray) and isinstance(y_pred, np.ndarray):
            return np.mean(y_true == y_pred)
        else:
            raise Exception('Error! Targets are not in numpy array!')


class PrecisionScore:
    def __call__(self, y_true, y_pred):
        if isinstance(y_real, np.ndarray) and isinstance(y_pred, np.ndarray):
            pass
        else:
            raise Exception('Error! Targets are not in numpy array!')


class RecallScore:
    def __call__(self, y_true, y_pred):
        if isinstance(y_real, np.ndarray) and isinstance(y_pred, np.ndarray):
            pass
        else:
            raise Exception('Error! Targets are not in numpy array!')


class F1Score:
    def __init__(self):
        self.precision = PrecisionScore()
        self.recall = RecallScore()

    def __call__(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)
