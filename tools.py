import csv
import os
import pandas as pd
import numpy as np

def sigmoid(z):
    eps1 = 1e-50
    eps2 = 1e+50
    
    #z = np.clip(z, eps1, eps2)
    #print(min(z), max(z))
    return 1.0 / (1.0 + np.exp(-z))

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