from tools import csv_load, sigmoid
from describe import mean_, std_
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.metrics import log_loss


class LogRe():
    def __init__(self):
        self.w = None
        self.bias = None
        self.epochs = 0
        self.l_rate = 0
        self.loss = []

    def train(self, l_rate, epochs):
        self.epochs = epochs
        self.l_rate = l_rate

        # preprocessing
        data = csv_load('datasets/dataset_train.csv')  
        data = data.dropna()
        data['Best Hand'] = data['Best Hand'].replace(['Right', 'Left'], [0, 1])
        x = data.loc[:, 'Best Hand':'Flying']
        # standardize
        for i in x:
            tmp_mean = mean_(x[i])
            tmp_std = std_(x[i], tmp_mean)
            x[i] = x[i].fillna(tmp_mean)
            x[i] = (x[i] - tmp_mean) / tmp_std
    
        # data split for multiclass classification
        new_y = pd.DataFrame()
        models = pd.DataFrame()
        y = data['Hogwarts House'].values
        labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        values = [[], [], [], []]
        self.loss = {k: v for k, v in zip(labels, values)}
        for i in labels:
            new_y[i] = y
            new_y.loc[new_y[i] == i, i] = 1
            new_y.loc[new_y[i] != 1, i] = 0

        # multiclass classification
        for j in new_y:
            LogRe.GD(self, x.values, new_y[j].values, j)
            model = pd.Series(self.w.reshape(-1).tolist())
            print(type(model))
            models[j] = model
        models = pd.DataFrame(models)
        models.to_csv('weights.csv', encoding='utf-8')
        models.to_pickle('weights.pkl')

    def GD(self, x, y, j):
        y = np.vstack(y)
        self.w = np.zeros((np.shape(x)[1], 1))
        m = np.shape(y)[0]  # number of rows
        self.bias = np.vstack(np.ones(np.shape(x)[0]))

        # propagation
        print('Training classifier on class', j)
        for i in range(0, self.epochs):
            #print(x.shape)
            z = np.dot(x, self.w) # (1600, 14) * (14, 1) = (1600, 1)
            h = sigmoid(z)
            residual = (h - y)
            grad_w = (self.l_rate / m) * (np.dot(x.T, residual)) + ((self.l_rate / m) * self.w) # (1600, 14) * (1600, 1) = (14, 1)
            grad_b = (self.l_rate / m) * np.dot(self.bias.T, residual)
            #h = np.clip(h, eps, 1 - eps)
            J = (1 / m) * (np.dot(-y.T, np.log(h)) - (np.dot((1 - y).T, np.log(1 - h)))) + sum((self.l_rate / (2*m)) * (self.w**2))
            if i == 0 or i % 10 == 0:
                print(i, '/ 100:', 'Loss', J[0][0], log_loss(y, h))
            self.loss[j].append(J[0][0])
            self.w = self.w - grad_w
            self.bias = self.bias - grad_b
        print(type(self.bias[0]))
        self.w = np.vstack((self.bias[0], self.w))
        print('')

    def GD_plot(self):
        for i in self.loss:
            plt.plot(np.arange(len(self.loss[i])), self.loss[i])
        plt.legend(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], loc='upper right')
        plt.title("Development of loss during training")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")
        plt.show()


    # https://jonchar.net/notebooks/Logistic-Regression/ 
    # print(self.w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.5, type=float)
    parser.add_argument('--viz', action="store_true")
    args = parser.parse_args()
    #print(args.__dict__)
    l_rate = args.__dict__['learning_rate']
    if l_rate > 1:
        l_rate = 0.5
        print('Warning! Too big learning rate. It was set to 0.5.')
    if l_rate < 0.000001:
        l_rate = 0.5
        print('Warning! Too small learning rate. It was set to 0.5.')
    l_rate = 0.1
    epochs = 100
    LR = LogRe()
    LR.train(l_rate, epochs)
    if args.__dict__['viz'] == True:
        LR.GD_plot()


if __name__ == '__main__':
    main()
