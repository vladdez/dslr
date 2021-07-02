import sys
import argparse
import numpy as np
import pandas as pd
from tools import csv_load
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

def median_(col):
    cleanedcol = []
    for i in col:
        if np.isnan(i):
            continue
        else:
            cleanedcol.append(i)
    sortedcol = sorted(cleanedcol)
    col_len = len(cleanedcol)
    index = (col_len - 1) // 2

    if (col_len % 2):
        return sortedcol[index]
    else:
        return (sortedcol[index] + sortedcol[index + 1]) / 2.0


def std_(col, mean):
    sum = 0
    for i in col:
        if not np.isnan(i):
            sum += (i - mean) ** 2
    return (sum / len(col)) ** 0.5

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
    col = col.dropna()
    col = col.sort_values()
    index = int(len(col - 1) * p)
    
    return col[index]

def describe(d):
    num_cols = []

    for column in d:
        try:
            d[column] = pd.to_numeric(d[column])
            num_cols.append(column)
        except:
            pass
    d2 = d[num_cols]
    count = []
    mean = []
    median = []
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
        median.append(median_(d2[column]))
        minx.append(min_(d2[column]))
        maxx.append(max_(d2[column]))
        q2.append(percentile_(d2[column], 0.25))
        q3.append(percentile_(d2[column], 0.5))
        q4.append(percentile_(d2[column], 0.75))
    measures = [['Count', 'Mean', 'Std', 'Median', 'Min', '25%', '50%', '75%', 'Max']]
    res = pd.DataFrame(columns = d2.columns, data = np.array([count, mean, std, median, minx, q2, q3, q4, maxx]))
    res = res.set_index(measures)
    for column in res:
        res[column] = res[column].map('{:12.6f}'.format)
    print(res)
    #print(np.median(d['Flying']), np.std(d['Flying']), np.max(d['Flying']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="input dataset")
    arg = parser.parse_args().dataset
    print(arg)
    describe(csv_load(arg))

