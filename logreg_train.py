from tools import csv_load
from describe import mean_, std_
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

class LogRe():
    def __init__(self):
        self.w = None
        self.epochs = 0
        self.l_rate = 0
    


    def train(self, l_rate, epochs):
        self.epochs = epochs
        self.l_rate = l_rate
        self.mean = []
        self.std = []
        data = csv_load('datasets/dataset_train.csv')
        x = data.loc[:, 'Arithmancy':'Flying']
        for i in x:
            tmp_mean = mean_(x[i])
            self.mean.append(tmp_mean)
            tmp_std = std_(x[i], tmp_mean)
            self.std.append(tmp_std)
            x[i] = x[i].fillna(tmp_mean)
            x[i] = (x[i] - tmp_mean) / tmp_std
        bias = np.ones(np.shape(x)[0])
        x.insert(loc=0, column='bias', value = bias)
        y = data['Hogwarts House'].values
        labels = data['Hogwarts House'].unique()
        #print(x)
        # data split for multiclass classification
        new_y = pd.DataFrame()
        models = pd.DataFrame()
        for i in labels:
            new_y[i] = y
            new_y.loc[new_y[i] == i, i] = 1
            new_y.loc[new_y[i] != 1, i] = 0
        
        #for i in new_y:
        model = LogRe.GD(self, x.values, new_y[i].values)
            #model[i] = model
        #print(model)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def GD(self, x, y):
        y = np.vstack(y)
        self.w = np.zeros((np.shape(x)[1], 1))
        m = np.shape(y)[0]
        print(np.shape(x), np.shape(self.w))
        for i in range(1, self.epochs):
            z = x.dot(self.w) # (1600, 14) * (14, 1) = (1600, 1)
            h = self.sigmoid(z)
            e = (h - y)
            grad = (1/m) * (x.T.dot(e)) # (1600, 14) * (1600, 1) = (14, 1)
            self.w = self.w - grad
            #print(grad)
            print(self.w)

        np.savetxt('weights.csv', self.w)
        
        
        
        

       #https://jonchar.net/notebooks/Logistic-Regression/ 
        #print(self.w)
    
    



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