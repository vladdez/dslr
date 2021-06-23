from tools import csv_load
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def histogram(data, subject):
    r = data.query('`Hogwarts House` == "Ravenclaw"')
    g = data.query('`Hogwarts House` == "Gryffindor"')
    h = data.query('`Hogwarts House` == "Hufflepuff"')
    s = data.query('`Hogwarts House` == "Slytherin"')
    plt.hist(g[subject], color='red', alpha=0.5)
    plt.hist(h[subject], color='yellow', alpha=0.5)
    plt.hist(r[subject], color='blue', alpha=0.5)
    plt.hist(s[subject], color='green', alpha=0.5)
    plt.legend(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], loc='upper right')
    plt.xlabel(subject)
    plt.ylabel('Number of student')
    plt.title('Checking the homogeneousity')
    plt.show()


if __name__ == '__main__':
    data  = csv_load('datasets/dataset_train.csv')
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default = "Arithmancy")
    args = parser.parse_args()
    histogram(data, args.__dict__['subject'])