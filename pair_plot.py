import seaborn as sns
import matplotlib.pyplot as plt
import src.tools as tools

if __name__ == '__main__':
    args = tools.parse_args_dataset()
    dataset = tools.read_dataset(args['dataset_path'])
    plt.tight_layout()
    sns.pairplot(dataset, hue='Hogwarts House')
    plt.savefig('imgs/pairplot.png')
    print('Pairplot saved to imgs/pairplot.png.')
