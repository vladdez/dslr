from tools import csv_load
import numpy as np
import matplotlib.pyplot as plt
import argparse

class LogRe():
    a : Tensor 
    
    def __init__(self):
        self.th0 = 0.0
        self.th1 = 0.0
        
    def train(self, l_rate, epochs):
        data = csv_load('datasets/dataset_train.csv')
        
        x = data.loc[:, 'Arithmancy':'Flying'].values
        y = data['Hogwarts House'].values
        print(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.5, type=float)
    args = parser.parse_args()
    # print(args.__dict__)
    l_rate = args.__dict__['learning_rate']
    if l_rate > 1:
        l_rate = 0.5
        print('Warning! Too big learning rate. It was set to 0.5.')
    if l_rate < 0.000001:
        l_rate = 0.5
        print('Warning! Too small learning rate. It was set to 0.5.')
    epochs = 100
    LR = LogRe()
    LR.train(l_rate, epochs)

if __name__ == '__main__':
    main()