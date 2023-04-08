import pandas as pd
import numpy as np

def load_stan_cs229(path):
    data = pd.read_csv(path)
    m = data.shape[0]

    y = np.asarray(data.y).reshape(m, 1)
    X = np.asarray(data.iloc[:, : -1])

    return X, y

def load_linreg_sample(path):
    data = pd.read_csv(path)
    data = data.dropna()

    y = np.asarray(data.y).reshape(data.shape[0], 1)
    data = data.drop(['y'], axis=1)
    
    X =  np.asarray(data)
    ones = np.repeat(1, X.shape[0]).reshape(X.shape[0], 1) # x0 = 1
    X = np.hstack((ones, X))

    return X, y

def load_abalone(path):
    data = pd.read_csv(path)
    data = data[(data.Type == 'M') | (data.Type == 'I')]

    y = np.asarray([1 if x == 'M' else 0 for x in data.Type])
    y = y.reshape(y.shape[0], 1)
    
    
    X = np.asarray(data.iloc[:, 1:])
    ones = np.repeat(1, X.shape[0]).reshape(X.shape[0], 1) # x0 = 1
    X = np.hstack((ones, X))

    return X, y