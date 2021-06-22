import sys
import os
import csv
import argparse
import numpy as np
import pandas as pd


def csv_load(name):
    data = []
    try:
        with open(name, 'r') as file:
            if os.stat(name).st_size != 0:
                file = csv.reader(file, delimiter=',')
                for row in file:
                    data.append(row)
            else:
                exit()
    except csv.Error as e:
        print(e)
    return np.array(data)

def describe(data):


    print(pd.DataFrame(data).describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="input dataset")
    arg = parser.parse_args().dataset
    print(arg)
    describe(csv_load(arg))

