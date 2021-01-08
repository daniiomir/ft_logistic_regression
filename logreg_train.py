import numpy as np
import src.tools as tools
from src.model import MultiClassLogisticRegression

if __name__ == '__main__':
    args = tools.parse_args_train()
    scaler = tools.StandartScaler()
    dataset = tools.read_dataset(args['dataset_path'])
    remove_features = ['Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures']
    selected_features = dataset.select_dtypes(include=[np.number]).columns.to_list() + ['Hogwarts House']
    X_y_df = dataset[selected_features]
    X_y_df = X_y_df.drop(remove_features, axis=1)
    X_y_df.dropna(inplace=True)
    X = X_y_df.drop(['Hogwarts House'], axis=1)
    y = X_y_df['Hogwarts House']
    X_scaled = scaler.fit_transform(X)
    model = MultiClassLogisticRegression(eta=0.1, n_iter=100, verbose=True)
    X_scaled, y = X_scaled.to_numpy(), y.to_numpy()
    model.fit(X_scaled, y)
    print(f'Accuracy for training part - {model.score(X_scaled, y)}')
    model.loss_plot()
    model.save_weights(args['save_weights_path'])
    tools.save(scaler, args['save_scaler_path'])
    tools.save(model, args['save_model_path'])
