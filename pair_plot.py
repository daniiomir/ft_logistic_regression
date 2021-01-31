import matplotlib.pyplot as plt
import src.tools as tools

if __name__ == '__main__':
    args = tools.parse_args_dataset()
    df = tools.read_dataset(args['dataset_path'])
    houses = ['Slytherin', 'Gryffindor', 'Ravenclaw', 'Hufflepuff']
    colors = ['green', 'red', 'blue', 'yellow']
    courses = df.columns[6:].to_list()
    _, axs = plt.subplots(12, 12, figsize=(25, 15), tight_layout=True)
    for row_course, row_plt in zip(courses, axs):
        for col_course, col_plt in zip(courses, row_plt):
            if row_course == col_course:
                for house, color in zip(houses, colors):
                    marks = df[row_course][df['Hogwarts House'] == house].dropna()
                    col_plt.hist(marks, color=color, alpha=0.5)
            else:
                for house, color in zip(houses, colors):
                    x = df[row_course][df['Hogwarts House'] == house]
                    y = df[col_course][df['Hogwarts House'] == house]
                    col_plt.scatter(x, y, color=color, alpha=0.5)

            col_plt.tick_params(labelbottom=False)
            col_plt.tick_params(labelleft=False)

            if col_plt.is_last_row():
                col_plt.set_xlabel(col_course.replace(' ', '\n'))

            if col_plt.is_first_col():
                label = row_course.replace(' ', '\n')
                length = len(label)
                if length > 14 and '\n' not in label:
                    label = label[:int(length / 2)] + '\n' + \
                            label[int(length / 2):]
                col_plt.set_ylabel(label)

    plt.legend(houses,
               loc='center left',
               frameon=False,
               bbox_to_anchor=(1, 0.5))

    plt.savefig('imgs/pairplot.png')
    print('Pairplot saved to imgs/pairplot.png.')
