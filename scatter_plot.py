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
    sns.scatterplot(x=houses_dfs['Slytherin']['Astronomy'], y=houses_dfs['Slytherin']['Defense Against the Dark Arts'], label='Slytherin', color='green')
    sns.scatterplot(x=houses_dfs['Gryffindor']['Astronomy'], y=houses_dfs['Gryffindor']['Defense Against the Dark Arts'], label='Gryffindor', color='red')
    sns.scatterplot(x=houses_dfs['Ravenclaw']['Astronomy'], y=houses_dfs['Ravenclaw']['Defense Against the Dark Arts'], label='Ravenclaw', color='blue')
    sns.scatterplot(x=houses_dfs['Hufflepuff']['Astronomy'], y=houses_dfs['Hufflepuff']['Defense Against the Dark Arts'], label='Hufflepuff', color='yellow')
    plt.legend()
    plt.savefig('imgs/scatter.png')
    print('Histograms saved to imgs/scatter.png.')