from . import predicters as pred
import numpy as np

class LogisticRegression(pred.Predicters):
    def __init__(self, trainX: np.ndarray, trainY: np.ndarray, alpha = 0.1, precision = 1e-5):
        assert trainX.shape[0] == trainY.shape[0]

        self.X = trainX
        self.y = trainY
        self.alpha = alpha / trainX.shape[0]
        self.precision = precision
        self.theta = np.zeros((trainX.shape[1], 1))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self) -> np.ndarray:
        m, n = self.X.shape
        last_gradient = np.zeros((n, 1))
        error = []
        while True:
            gradient = (self.X.T @ (self.y - LogisticRegression.sigmoid(self.X @ self.theta).reshape(m, 1)))
            self.theta = self.theta + self.alpha * gradient
            
            error.append( np.mean((gradient - last_gradient) ** 2) )
            last_gradient = gradient

            if error[-1] < self.precision:
                break
        return error
    
    def fit_newt(self) -> np.ndarray:
        m, n = self.X.shape

        self.theta = np.zeros((n, 1))
        theta_last = np.zeros((n, 1))
        W = np.zeros((m, m))

        error = []
        while True:
            sigs = LogisticRegression.sigmoid(self.X @ self.theta)
            np.fill_diagonal(W, (sigs * (1 - sigs)))

            z = self.X @ self.theta + np.linalg.solve(W, self.y - sigs)

            theta_last = self.theta
            self.theta = np.linalg.solve(self.X.T @ W @ self.X, self.X.T @ W @ z)

            error.append(np.mean(((theta_last - self.theta) ** 2)))            
            if error[-1] < self.precision:
                break
        return error

    def predict(self, x: np.ndarray) -> int:
        return 0 if LogisticRegression.sigmoid(self.theta.T @ x) < 0.5 else 1
