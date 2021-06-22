import sys
import os
import csv
import argparse
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

def count_(col):
    return len(col)

def mean_(col):
    sum = 0
    for i in col:
        if np.isnan(i):
            continue
        sum += i
    return sum / len(col)
    
def std_(col, mean):
    sum = 0
    for i in col:
        if np.isnan(i):
            continue
        sum += i - mean
    return sum / len(col)

def min_(col):
    m = col[0]
    for i in col:
        if m > i:
            m = i
    return m

def max_(col):
    m = col[0]
    for i in col:
        if m < i:
            m = i
    return m

def percentile_(col, p):
    col = col.sort_values()
    index = int(len(col - 1) * p)
    
    return col[index]

def csv_load(name):
    data = []
    try:
        with open(name, 'r') as file:
            if os.stat(name).st_size != 0:
                file = csv.reader(file, delimiter=',')
                for _ in file:  
                    row = []
                    for value in _:
                        try:
                            value = float(value)
                        except:
                            pass
                            #if not value:
                              #  value = np.nan
                        row.append(value)
                    data.append(row)
            else:
                exit()
    except csv.Error as e:
        print(e)
    return np.array(data)

def describe(data):
    num_cols = []
    d = pd.DataFrame(data,  columns=data[0])
    d = d.drop(d.index[0]).reset_index()
    d = d.drop('index', axis='columns')
    for column in d:
        try:
            #d[column] = d[column].dropna()
            d[column] = pd.to_numeric(d[column])
            
            num_cols.append(column)
        except:
            pass
            #print(d[column].isna().sum() )
    d2 = d[num_cols]
    count = []
    mean = []
    std = []
    minx = []
    maxx = []
    q2 = []
    q3 = []
    q4 = []
    for column in d2:
        count.append(count_(d2[column]))
        m = mean_(d2[column])
        mean.append(m)
        std.append(std_(d2[column], m))
        minx.append(min_(d2[column]))
        maxx.append(max_(d2[column]))
        q2.append(percentile_(d2[column], 0.25))
        q3.append(percentile_(d2[column], 0.5))
        q4.append(percentile_(d2[column], 0.75))
    measures = [['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    res = pd.DataFrame(columns = d2.columns, data = np.array([count, mean, std, minx, q2, q3, q4, maxx]))
    res = res.set_index(measures)
    for column in res:
        res[column] = res[column].map('{:12.6f}'.format)
    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="input dataset")
    arg = parser.parse_args().dataset
    print(arg)
    describe(csv_load(arg))

