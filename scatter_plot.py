from tools import csv_load
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def scatter_plot(data, subject1, subject2):
    plt.scatter(data[subject1], data[subject2], alpha=0.5)
    plt.xlabel(subject1)
    plt.ylabel(subject2)
    plt.title('Checking similarity of features')
    plt.show()

if __name__ == '__main__':
    data  = csv_load('datasets/dataset_train.csv')
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject1", default = "Astronomy")
    parser.add_argument("--subject2", default = "Defense Against the Dark Arts")
    args = parser.parse_args()
    scatter_plot(data, args.__dict__['subject1'], args.__dict__['subject2'])