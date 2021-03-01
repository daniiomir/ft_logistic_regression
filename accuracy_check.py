import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    pred = pd.read_csv("houses.csv", delimiter=",")
    true = pd.read_csv("datasets/dataset_truth.csv", delimiter=",")

    y_pred = pred['Hogwarts House']
    y_true = true['Hogwarts House']
    print(accuracy_score(y_true, y_pred))
