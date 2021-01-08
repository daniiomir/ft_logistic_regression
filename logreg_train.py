import numpy as np
import src.tools as tools
from src.model import MultiClassLogisticRegression

if __name__ == '__main__':
    args = tools.parse_args_train()
    scaler = tools.StandartScaler()
    dataset = tools.read_dataset(args['dataset_path'])
    remove_features = ['Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures']
    dataset = dataset.drop(remove_features, axis=1)
    selected_features = dataset.select_dtypes(include=[np.number]).columns + ['Hogwarts House']
    X_y_df = dataset[selected_features]
    X = X_y_df.drop(['Hogwarts House'], axis=1)
    y = X_y_df['Hogwarts House']
    X_scaled = scaler.fit_transform(X)
    model = MultiClassLogisticRegression(eta=0.1, solver='', n_iter=10000, verbose=True)
    model.fit(X_scaled, y)
    print(f'Accuracy for training part - {model.score(X_scaled, y)}')
    model.loss_plot()
    model.save_weights(args['save_model_path'])
    tools.save(scaler, args['save_scaler_path'])
