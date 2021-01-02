import pandas as pd
import numpy as np


def select_features(dataset, numeric=True):
    if numeric:
        return dataset.select_dtypes(include=[np.number])
    return dataset.select_dtypes(include=['object'])


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
