import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import src.tools as tools

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    args = tools.parse_args_dataset()
    dataset = tools.read_dataset(args['dataset_path'])
    houses = ['Slytherin', 'Gryffindor', 'Ravenclaw', 'Hufflepuff']
    houses_dfs = {}
    sns.set_style('darkgrid')
    for house in houses:
        df = dataset[dataset['Hogwarts House'] == house]
        houses_dfs[house] = tools.select_features(df, dropna=True)
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    fig.suptitle('All features distribution')
    fig.tight_layout()
    for index, feature in enumerate(tools.select_features(dataset).columns):
        plt.subplot(4, 4, index + 1)
        sns.distplot(houses_dfs['Slytherin'][feature], label='Slytherin', bins=25, color='green')
        sns.distplot(houses_dfs['Gryffindor'][feature], label='Gryffindor', bins=25, color='red')
        sns.distplot(houses_dfs['Ravenclaw'][feature], label='Ravenclaw', bins=25, color='blue')
        sns.distplot(houses_dfs['Hufflepuff'][feature], label='Hufflepuff', bins=25, color='yellow')
        plt.legend()
    plt.savefig('imgs/hist.png')
    print('Histograms saved to imgs/hist.png.')
