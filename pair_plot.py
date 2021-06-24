from tools import csv_load
import matplotlib.pyplot as plt
import seaborn as sns


def pair_plot(data):
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(data)
    plt.show()

if __name__ == '__main__':
    data = csv_load('datasets/dataset_train.csv')
    pair_plot(data.loc[:, 'Arithmancy':'Flying'])