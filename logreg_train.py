from tools import csv_load
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
        data = csv_load('datasets/dataset_train.csv')
        
        x = data.loc[:, 'Arithmancy':'Flying']
        bias = np.ones(np.shape(x)[0])
        x.insert(loc=0, column='bias', value = bias)
        y = data['Hogwarts House'].values
        labels = data['Hogwarts House'].unique()

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
        self.w = np.zeros(np.shape(x))
        w = np.zeros(np.shape(x))
        m = np.shape(y)[0]
        print(x)
        print(y)
        for i in range(1, self.epochs):
            z = x * w 
            #print(z)
            h = self.sigmoid(z)
            #print('h', h)
            e = (h.T - np.array(y))
            #print(e)
            derivated_loss = (self.l_rate/m) * (x.T * e)
            #print(derivated_loss)
            self.w = self.w - derivated_loss.T
            print(self.w.shape)
            #a = self.w  * x
            #a = LogRe.sigmoid(self, self.w  * x)
            #J = np.log(a).T
            #J = -(1/m) * (np.log(LogRe.sigmoid(self, x * self.w)).T * y + np.log(1 - LogRe.sigmoid(self, x * self.w)).T * (1 - y))
            #print(J)
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
    epochs = 2
    LR = LogRe()
    LR.train(l_rate, epochs)

if __name__ == '__main__':
    main()