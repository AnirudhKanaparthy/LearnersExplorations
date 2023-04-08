import algorithms.predicters as pred
import algorithms.linear_regression as linreg
import numpy as np

def classification_test(model: pred.Predicters, testX: np.ndarray, testY: np.ndarray) -> float:
    m = testX.shape[0]
    res = 0
    for i in range(m):
        res += int(model.predict(testX[i]) == testY[i])
    return res / m * 100

def regression_test(model: linreg.LinearRegression, testX: np.ndarray, testY: np.ndarray) -> float:
    predictions = testX @ model.theta

    mse = np.mean((predictions - testY) ** 2)
    return mse 