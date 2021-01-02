import pickle
import argparse
import pandas as pd


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
