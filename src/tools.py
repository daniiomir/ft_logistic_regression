import pickle
import argparse
import pandas as pd
import numpy as np


class StandartScaler:
    def __init__(self):
        self.columns = []
        self.mean = {}
        self.std = {}

    def _fit(self, data):
        for col in data.columns:
            self.mean[col] = np.mean(data[col].values)
            self.std[col] = np.std(data[col].values, ddof=1)
        self.columns = data.columns.to_list()
        self.columns.sort()

    def _scale(self, value, mean, std):
        return (value - mean) / std

    def fit_transform(self, data):
        if isinstance(data, pd.DataFrame):
            self._fit(data)
            return self.transform(data)
        raise Exception('Passed argument should be pandas dataframe.')

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            cols = data.columns.to_list()
            cols.sort()
            if cols == self.columns:
                data = data.copy()
                for col in data.columns:
                    data[col] = data[col].apply(self._scale, mean=self.mean[col], std=self.std[col])
                return data
            raise Exception('Dataframe columns are not equal to previous.')
        raise Exception('Passed argument should be pandas dataframe.')


def select_features(dataset, numeric=True, dropna=False):
    if numeric:
        df = dataset.select_dtypes(include=[np.number])
    else:
        df = dataset.select_dtypes(include=['object'])
    if dropna:
        df = df.dropna()
    return df


def describe(dataset):
    columns = [' ', 'Count', 'Nan', 'Not nan', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    df = pd.DataFrame(columns=columns)
    dataset = select_features(dataset)
    for column in dataset.columns:
        user_dict = {i: 0 for i in columns}
        column_array = np.sort(dataset[column].values)
        user_dict[' '] = column
        user_dict['Count'] = column_array.shape[0]
        user_dict['Nan'] = np.count_nonzero(np.isnan(column_array))
        user_dict['Not nan'] = np.count_nonzero(~np.isnan(column_array))
        column_array = column_array[~np.isnan(column_array)]
        user_dict['Mean'] = np.mean(column_array)
        user_dict['Std'] = np.std(column_array, ddof=1)
        user_dict['Min'] = column_array.astype(np.float64).min()
        user_dict['25%'] = np.percentile(column_array, q=25, interpolation='nearest')
        user_dict['50%'] = np.percentile(column_array, q=50, interpolation='nearest')
        user_dict['75%'] = np.percentile(column_array, q=75, interpolation='nearest')
        user_dict['Max'] = column_array.astype(np.float64).max()
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
    parser.add_argument('--save_model_path', default='model.pkl')
    parser.add_argument('--save_scaler_path', default='scaler.pkl')
    args = parser.parse_args()
    return args.__dict__


def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='datasets/dataset_test.csv')
    parser.add_argument('--load_model_path', default='model.pkl')
    parser.add_argument('--load_scaler_path', default='scaler.pkl')
    args = parser.parse_args()
    return args.__dict__


def read_dataset(path):
    dataset = pd.read_csv(path, index_col='Index')
    return dataset


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, path, protocol=4)


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
