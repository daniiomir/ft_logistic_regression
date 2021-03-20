import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import src.tools as tools
from src.model import LogisticRegression, OneVSAllClassifier

if __name__ == '__main__':
    args = tools.parse_args_train()
    scaler = tools.StandartScaler()
    encoder = tools.LabelEncoder()

    dataset = tools.read_dataset(args['dataset_path'])

    corr = dataset.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns, annot=True)
    plt.tight_layout()
    plt.savefig('imgs/corr_plot.png')
    plt.close()

    remove_features = ['Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures', 'Flying']
    selected_features = dataset.select_dtypes(include=[np.number]).columns.to_list() + ['Hogwarts House', 'Best Hand']
    X_y_df = dataset[selected_features]
    X_y_df = X_y_df.drop(remove_features, axis=1)
    X_y_df.dropna(inplace=True)

    X_num = X_y_df.drop(['Hogwarts House', 'Best Hand'], axis=1)
    X_num = scaler.fit_transform(X_num)
    X_cat = pd.get_dummies(X_y_df['Best Hand'])
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.to_numpy()
    y = X_y_df['Hogwarts House']

    y = encoder.fit_transform(y)

    model_args = {
        'eta': 0.01,
        'n_iter': 100,
        'verbose': True,
        'verbose_epoch': 10,
        'decision_thres': 0.5,
        'l1_ratio': 0.1
    }

    if args['clf'] == 'multiclass':
        model = LogisticRegression(multiclass=True, **model_args)
    elif args['clf'] == 'ova':
        model = OneVSAllClassifier(algo=LogisticRegression, **model_args)
    else:
        raise NotImplementedError

    model.fit(X, y)
    print(f'Accuracy for training part - {model.score(X, y)}')

    model.loss_plot()
    tools.save((scaler, encoder), args['save_tools_path'])
    tools.save(model, args['save_model_path'])
