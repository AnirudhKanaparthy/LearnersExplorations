import predicters as pred
import numpy as np

class LinearRegression(pred.Predicters):
    def __init__(self, trainX: np.ndarray, trainY: np.ndarray):
        assert trainX.shape[0] == trainY.shape[0]

        self.X = trainX
        self.y = trainY
        self.theta = np.zeros((trainX.shape[1], 1))

    def lms(self) -> float:
        return 0.5 * np.sum((self.X @ self.theta - self.y) ** 2)

    def fit(self, alpha = 0.001, threshold = 0.01, max_iter = int(10e4)):
        m, n = self.X.shape
        alpha / ( 2 * self.X.shape[0] )

        gradient = np.zeros((n, 1))

        for i in range(1, max_iter):
            gradient = self.X.T @ (self.X @ self.theta - self.y)
            self.theta = self.theta - (alpha) * gradient
            gra = (gradient.T @ gradient)[0][0]

            if i % 1000 == 0:
                print(gra)
            if gra < threshold:
                break
    
    def fit_newt(self):
        self.theta = np.linalg.solve(self.X.T @ self.X, self.X.T @ self.y)

    def predict(self, x: np.ndarray) -> float:
        return self.theta.T @ x
