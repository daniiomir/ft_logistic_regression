import numpy as np
import src.tools as tools
from src.model import LogisticRegression, OneVSAllClassifier

if __name__ == '__main__':
    args = tools.parse_args_train()
    scaler = tools.StandartScaler()
    encoder = tools.LabelEncoder()

    dataset = tools.read_dataset(args['dataset_path'])

    remove_features = ['Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures']
    selected_features = dataset.select_dtypes(include=[np.number]).columns.to_list() + ['Hogwarts House']
    X_y_df = dataset[selected_features]
    X_y_df = X_y_df.drop(remove_features, axis=1)
    X_y_df.dropna(inplace=True)
    X = X_y_df.drop(['Hogwarts House'], axis=1)
    y = X_y_df['Hogwarts House']

    X_scaled = scaler.fit_transform(X).to_numpy()
    y = encoder.fit_transform(y)

    if args['clf'] == 'multiclass':
        model = LogisticRegression(eta=0.001, multiclass=True, n_iter=100, verbose=True, verbose_epoch=1)
    elif args['clf'] == 'ova':
        model = OneVSAllClassifier(algo=LogisticRegression, eta=0.1, n_iter=50, verbose=True, verbose_epoch=1)
    else:
        raise NotImplementedError

    model.fit(X_scaled, y)
    print(f'Accuracy for training part - {model.score(X_scaled, y)}')

    model.loss_plot()
    tools.save((scaler, encoder), args['save_tools_path'])
    tools.save(model, args['save_model_path'])
