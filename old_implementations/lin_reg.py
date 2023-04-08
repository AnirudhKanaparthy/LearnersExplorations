import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "../datasets/cars.csv"

dataset = pd.read_csv(file_path)

#features = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight']
features = ['Horsepower']

X = dataset[features].to_numpy()
#y = dataset.Acceleration.to_numpy()[:, np.newaxis]
y = dataset.Weight.to_numpy()[:, np.newaxis]


# Portions we want to use for training and for testing.
percent_train = 50
portion = int((percent_train/100) * len(X))


# Dividing the data into training and testing portions.
train_X = X[ :portion]
val_X = X[portion: ]

train_y = y[ : portion]
val_y = y[portion: ]

global_error = list()

def global_error_plot():
    plt.plot(range(len(global_error)), global_error)
    plt.show()

def compute_cost(X, y, params):
    n_samples = len(y)
    h = X @ params
    return (1/(2*n_samples)) * np.sum((h-y)**2)


## Newton's Method optimization ##
def newt_gra_des(X, y, learning_rate = 0.1, tolerance = 0.000001):
    m = len(y)
    param = np.zeros((len(X[0]), 1))
    
    while True:
        hess_inv = np.linalg.inv(X.T @ X)

        new_param = param - hess_inv @ X.T @ (X @ param - y)
        global_error.append(compute_cost(X, y, param))
        if np.all(abs(new_param - param) <= tolerance):
            print("Converged.")
            break
        param = new_param
    return param

def gradient_descent(X, y, learning_rate = 0.1, tolerance = 0.0000001):
    m = len(y)
    param = np.zeros((len(X[0]), 1))
    
    while True:
        new_param = param - (learning_rate/(2*m)) * X.T @ (X @ param - y)
        global_error.append(compute_cost(X, y, param))
        if np.all(abs(new_param - param) <= tolerance):
            print("Converged.")
            break
        param = new_param
    return param

def mean_squared_error(target_y, pred_y):
    m = len(target_y)
    return (1/m) * np.sum((target_y - pred_y)**2)

def predict(X, params):
    mat_X = X[np.newaxis, :]
    prediction = mat_X @ params
    return prediction

## Normalizing data ##
train_X = (train_X - np.mean(train_X, 0)) / np.std(train_X, 0)
train_X = np.hstack((np.ones((len(train_X), 1)), train_X))

params = gradient_descent(train_X, train_y, 0.00025)
#params = newt_gra_des(train_X, train_y, 0.00025)

## Normalizing data ##
val_X = (val_X - np.mean(val_X, 0)) / np.std(val_X, 0)
val_X = np.hstack((np.ones((len(val_X), 1)), val_X))

predictions = val_X @ params

msq_error = mean_squared_error(val_y, predictions)

## Output ##
print(dataset.head(), "\n")
print("Final params: ", params.T)
print("Final Error:", global_error[-1])
print("Mean Squared Error:", msq_error)

def plot():
    plt_X = val_X.T[1]
    plt_y = val_y.T[0]
    
    plt.plot(plt_X, predictions, color = 'r')
    plt.scatter(plt_X, plt_y)
    
    plt.show()

plot()
global_error_plot()