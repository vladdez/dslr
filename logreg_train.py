from tools import csv_load
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse

class LiRe():
    def __init__(self):
        self.th0 = 0.0
        self.th1 = 0.0
        self.milages = []
        self.prices = []
        self.milages_n = []
        self.prices_n = []
        self.p_mean = 0
        self.p_std = 0
        self.evolution = [[0.0, 0.0]]

    def get_data(self):
        data = csv_load('datasets/dataset_train.csv')

    def EstimatePrice(self, milage, th0, th1):
        return th0 + th1 * milage

    def GD(self, l_rate, epochs):
        self.prices_n = (self.prices - self.p_mean) / self.p_std
        length = len(self.prices)
        for i in range(0, epochs - 1):
            tmp0 = 0.0
            tmp1 = 0.0
            # mean squared error
            for milage, price in zip(self.milages_n, self.prices_n):
                h = LiRe.EstimatePrice(self, milage, self.th0, self.th1)
                error = h - price
                tmp0 += error
                tmp1 += error * milage
            self.th0 -= l_rate * (tmp0 / length)
            self.th1 -= l_rate * (tmp1 / length)
            if i % 10 == 0:
                a, b = LiRe.denormalization(self)
                self.evolution.append([a + b *100, a + b * 250000])
        self.th0, self.th1 = LiRe.denormalization(self)
        data = np.column_stack([self.th0, self.th1])
        print("{:.2f}, {:.2f}".format(self.th0, self.th1))
        np.savetxt('thetas.csv', data, fmt=['%f', '%f'])

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
    LR = LiRe()
    LR.get_data()
    LR.GD(l_rate, epochs)

if __name__ == '__main__':
    main()