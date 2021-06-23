from tools import csv_load
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns


def pair_plot(data):
    sns.pairplot(df, hue='class')  
    plt.show()

if __name__ == '__main__':
    data  = csv_load('datasets/dataset_train.csv')
    parser = argparse.ArgumentParser()
    #parser.add_argument("--subject1", default = "Astronomy")
    #parser.add_argument("--subject2", default = "Defense Against the Dark Arts")
    #args = parser.parse_args()
    #scatter_plot(data, args.__dict__['subject1'], args.__dict__['subject2'])
    pair_plot(data.loc[:, 'Arithmancy':'Flying'])