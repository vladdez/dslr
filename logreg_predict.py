import pandas as pd
import numpy as np
from tools import sigmoid
from describe import mean_, std_

if __name__ == '__main__':
    w = pd.read_pickle('weights.pkl').to_numpy()
   
    X = pd.read_csv('datasets/dataset_test.csv')
    bias = pd.DataFrame(np.ones(np.shape(X)[0]))
    hand = X['Best Hand'].replace(['Right', 'Left'], [0, 1])
    X = X.loc[:, 'Arithmancy':'Flying']
   # for i in X:
   #     X[i] = X[i].fillna(0) #?????
    for i in X.loc[:, 'Arithmancy':'Flying']:
        tmp_mean = mean_(X[i])
        tmp_std = std_(X[i], tmp_mean)
        X[i] = X[i].fillna(tmp_mean)
        X[i] = (X[i] - tmp_mean) / tmp_std
    X = pd.concat([hand, X], axis=1)
    X = pd.concat([bias, X], axis=1)
    print(pd.DataFrame(w))
    X = X.to_numpy()

    print(X.shape, w.shape)
    h = np.dot(X, w)
    
    pred = sigmoid(h)  # (400, 14) (14, 4) = (400, 4)
    #print(pred)
    num_res = np.argmax(pred, axis=1)
    legend = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}
    houses = []
    for i in num_res:
        houses.append(legend[i])
    preds_df = pd.DataFrame(data=houses, columns=['Hogwarts House'])
    preds_df.index.name = 'Index'
    preds_df.to_csv('houses.csv')
    print('Prediction saved to houses.csv')
    print(preds_df.value_counts(['Hogwarts House']))
