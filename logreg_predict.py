import numpy as np
import src.tools as tools
from src.model import MultiClassLogisticRegression

if __name__ == '__main__':
    args = tools.parse_args_test()
    scaler = tools.load(args['load_scaler_path'])
    dataset = tools.read_dataset(args['dataset_path'])
    remove_features = ['Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures']
    dataset = dataset.drop(remove_features, axis=1)
    selected_features = dataset.select_dtypes(include=[np.number]).columns
    X_y_df = dataset[selected_features]
    X = X_y_df.drop(['Hogwarts House'], axis=1)
    X_scaled = scaler.fit_transform(X)
    model = MultiClassLogisticRegression()
    model.load_weights(args['load_model_path'])
    preds = model.predict_classes(X_scaled)
