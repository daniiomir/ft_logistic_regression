import pickle
import argparse
import pandas as pd
import numpy as np


class StandartScaler:
    def __init__(self):
        self.mean = {}
        self.std = {}

    def _fit(self, data):
        for col in data.columns:
            self.mean[col] = np.mean(data[col].values)
            self.std[col] = np.std(data[col].values, ddof=1)

    def _scale(self, value, mean, std):
        return (value - mean) / std

    def fit_transform(self, data):
        self._fit(data)
        return self.transform(data)

    def transform(self, data):
        for col in data.columns:
            data[col] = data[col].apply(self._scale, mean=self.mean[col], std=self.std[col])
        return data


def parse_args_describe():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='datasets/dataset_train.csv')
    args = parser.parse_args()
    return args.__dict__


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='datasets/dataset_train.csv')
    parser.add_argument('--save_model_path', default='model.pkl')
    args = parser.parse_args()
    return args.__dict__


def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='datasets/dataset_test.csv')
    parser.add_argument('--load_model_path', default='model.pkl')
    args = parser.parse_args()
    return args.__dict__


def read_dataset(path):
    dataset = pd.read_csv(path, index_col='Index')
    return dataset


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=4)


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
