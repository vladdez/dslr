import csv
import os
import pandas as pd

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