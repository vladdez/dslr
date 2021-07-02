import csv
import os
import pandas as pd
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    print(np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1))
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

def csv_load(name):
    try:
        d = pd.read_csv(name)
    except FileNotFoundError as e:
        print(e)
        exit()
    except pd.errors.EmptyDataError as e:
        print(e)
        exit()
    return d