import pickle
import argparse
import pandas as pd
import numpy as np


class LabelEncoder:
    def __init__(self):
        self.mapping = None

    def _fit(self, targets: np.ndarray):
        classes = np.unique(targets)
        self.mapping = {classes[i]: i for i in range(len(classes))}

    def fit_transform(self, targets: np.ndarray):
        self._fit(targets)
        return self.transform(targets)

    def transform(self, targets: np.ndarray):
        if self.mapping is not None:
            encoded = np.zeros(targets.shape, dtype=int)
            for k, v in self.mapping.items():
                encoded[targets == k] = v
            return encoded
        raise Exception('You should do fit_transform first!')

    def reverse_transform(self, reversed_targets: np.ndarray):
        if self.mapping is not None:
            targets = np.zeros(reversed_targets.shape, dtype=object)
            for k, v in self.mapping.items():
                targets[reversed_targets == v] = k
            return targets
        raise Exception('You should do fit_transform first!')


class StandartScaler:
    def __init__(self):
        self.columns = []
        self.mean = {}
        self.std = {}

    def _fit(self, data: pd.DataFrame):
        for col in data.columns:
            self.mean[col] = np.mean(data[col].values)
            self.std[col] = np.std(data[col].values, ddof=1)
        self.columns = data.columns.to_list()
        self.columns.sort()

    def _scale(self, value, mean, std):
        return (value - mean) / std

    def fit_transform(self, data: pd.DataFrame):
        self._fit(data)
        return self.transform(data)

    def transform(self, data: pd.DataFrame):
        cols = data.columns.to_list()
        cols.sort()
        if cols == self.columns:
            data = data.copy()
            for col in data.columns:
                data[col] = data[col].apply(self._scale, mean=self.mean[col], std=self.std[col])
            return data
        raise Exception('Dataframe columns are not equal to previous.')


def one_hot_encoding(y):
    labels = np.unique(y)
    mapping = {labels[i]: i for i in range(len(labels))}
    return np.eye(len(labels))[np.vectorize(lambda c: mapping[c])(y).reshape(-1)]


def select_features(dataset: pd.DataFrame, numeric: bool = True, dropna: bool = False):
    if numeric:
        df = dataset.select_dtypes(include=[np.number])
    else:
        df = dataset.select_dtypes(include=['object'])
    if dropna:
        df = df.dropna()
    return df


def _mean(x: np.ndarray):
    return np.sum(x) / x.shape[0]


def _std(x: np.ndarray):
    mean = _mean(x)
    total = 0
    for _ in x:
        total += (_ - mean) ** 2
    return (total / (x.shape[0] - 1)) ** 0.5


def _min(x: np.ndarray):
    min_value = x[0]
    for _ in x:
        if _ < min_value:
            min_value = _
    return min_value


def _max(x: np.ndarray):
    max_value = x[0]
    for _ in x:
        if _ > max_value:
            max_value = _
    return max_value


def _percentile(x, p):
    k = (x.shape[0] - 1) * (p / 100)
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return x[int(k)]
    d0 = x[int(f)] * (c - k)
    d1 = x[int(c)] * (k - f)
    return d0 + d1


def describe(dataset: pd.DataFrame):
    columns = [' ', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    df = pd.DataFrame(columns=columns)
    dataset = select_features(dataset)
    for column in dataset.columns:
        if np.isnan(dataset[column]).all():
            continue
        user_dict = {i: 0 for i in columns}
        column_array = np.sort(dataset[column].values)
        user_dict[' '] = column
        user_dict['Count'] = column_array.shape[0]
        column_array = column_array[~np.isnan(column_array)]
        user_dict['Mean'] = _mean(column_array)
        user_dict['Std'] = _std(column_array)
        user_dict['Min'] = _min(column_array.astype(np.float64))
        user_dict['25%'] = _percentile(column_array, 25)
        user_dict['50%'] = _percentile(column_array, 50)
        user_dict['75%'] = _percentile(column_array, 75)
        user_dict['Max'] = _max(column_array.astype(np.float64))
        df = pd.concat([df, pd.DataFrame([user_dict])])
    return df.set_index(' ').T.to_string()


def parse_args_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='datasets/dataset_train.csv')
    args = parser.parse_args()
    return args.__dict__


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='datasets/dataset_train.csv')
    parser.add_argument('--clf', default='ova')
    parser.add_argument('--save_model_path', default='tmp/model.pkl')
    parser.add_argument('--save_tools_path', default='tmp/tools.pkl')
    args = parser.parse_args()
    return args.__dict__


def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='datasets/dataset_test.csv')
    parser.add_argument('--load_model_path', default='tmp/model.pkl')
    parser.add_argument('--load_tools_path', default='tmp/tools.pkl')
    args = parser.parse_args()
    return args.__dict__


def read_dataset(path: str):
    dataset = pd.read_csv(path, index_col='Index')
    return dataset


def save(obj: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
