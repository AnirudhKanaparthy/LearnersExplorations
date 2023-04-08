import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# True means Logistic Regression
# False means Locally Weighted Logistic Regression
modelType = False


file_path = '../datasets/iris.csv'

dataset = pd.read_csv(file_path)

dataset = dataset.sample(frac = 1)

#features = ['sepal_length_in_cm', 'sepal_width_in_cm', 'petal_length_in_cm', 'petal_width_in_cm', 'flower_type']
features = ['sepal_length_in_cm']

X = dataset[features].to_numpy()
y = dataset.flower_type.to_numpy()[:, np.newaxis]


# Portions we want to use for training and for testing.
percent_train = 50
portion = int((percent_train/100) * len(X))


# Dividing the data into training and testing portions.
train_X = X[ :portion]
val_X = X[portion: ]

train_y = y[ : portion]
val_y = y[portion: ]

global_error = list()

def sigmoid(z):
    return 1/(1 + math.exp(-z))
sgmd = np.vectorize(sigmoid)

def global_error_plot():
    plt.plot(range(len(global_error)), global_error)
    plt.show()

def compute_cost(X, y, params):
    n_samples = len(y)
    h = sgmd(X @ params)
    return (1/(2*n_samples)) * np.sum((h-y)**2)

def weight(x_tar, x_val, tau):
    print(math.exp(-(np.sum(((x_tar - x_val)**2)**0.5))/(2*(tau**2))))
    return math.exp(-(np.sum(((x_tar - x_val)**2)**0.5))/(2*(tau**2)))

def weight_vector(X, x_target, tau = 0.01):
    weight_list = list()
    for datapoint in X:
        weight_list.append( weight(x_target, datapoint, tau) )
    return weight_list

def lwlr(X, y, x_target, learning_rate = 0.1, lam = 0.01, tolerance = 0.000001):
    m = len(y)
    param = np.zeros((len(X[0]), 1))
    weight_diag = np.diag(weight_vector(X, x_target, 0.07))
    
    while True:
        h = sgmd(X @ param)
        D = -weight_diag @ np.diag(h.T[0]) @ (1 - h)
        z = weight_diag @ (y - h)

        gradient = X.T @ z - lam*param

        hessian = X.T @ np.diag(D.T[0]) @ X - lam*(np.diag([1] * len(X[0])))
        hess_inv = np.linalg.inv(hessian)

        new_param = param - hess_inv @ gradient
        global_error.append(compute_cost(X, y, param))
        if np.all(abs(new_param - param) <= tolerance):
            print("Converged.")
            break
        param = new_param

    prediction = sgmd(x_target.T @ param)
    return prediction


def gradient_ascent(X, y, learning_rate = 0.1, tolerance = 0.000001):
    m = len(y)
    param = np.zeros((len(X[0]), 1))

    while True:
        h = sgmd(X @ param)
        new_param = param + (learning_rate/(2*m)) * X.T @ (y - h)
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


if modelType:
    params = gradient_ascent(train_X, train_y)


## Normalizing data ##
val_X = (val_X - np.mean(val_X, 0)) / np.std(val_X, 0)
val_X = np.hstack((np.ones((len(val_X), 1)), val_X))

lwlr_pred = list()
for inp in val_X:
    lwlr_pred.append(lwlr(train_X, train_y, inp))

plt_X = val_X.T[1]
plt_y = val_y.T[0]

plt.scatter(plt_X, lwlr_pred, color = 'r')
plt.scatter(plt_X, plt_y)
plt.show()



if modelType:
    predictions = sgmd(val_X @ params)

    msq_error = mean_squared_error(val_y, predictions)

    ## Output ##
    print(dataset.head(), "\n")
    print("Final params: ", params.T)
    print("Final Error:", global_error[-1])
    print("Mean Squared Error:", msq_error)
    
    plt_X = val_X.T[1]
    plt_y = val_y.T[0]

def plot():
    plt_pred = predictions
    
    plt.scatter(plt_X, plt_pred, color = 'r')
    plt.scatter(plt_X, plt_y)
    plt.show()

if modelType:
    plot()
    global_error_plot()