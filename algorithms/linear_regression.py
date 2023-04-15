from . import predicters as pred
import numpy as np

class LinearRegression(pred.Predicters):
    def __init__(self, trainX: np.ndarray, trainY: np.ndarray):
        assert trainX.shape[0] == trainY.shape[0]

        self.X = trainX
        self.y = trainY
        self.theta = np.zeros((trainX.shape[1], 1))


    def mse_train(self) -> float:
        return np.mean((self.X @ self.theta - self.y) ** 2)


    """
        The ultimate goal is to decrease the MeanSquaredError of the test data.
    """
    def mse_test(self, testX, testY) -> float:
        return np.mean((testX @ self.theta - testY) ** 2)


    """
        TODO: Stochastic Gradient Descent
        This uses batch gradient descent.
    """
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
    
    """
        You can think of Linear regression as finding the Theta for:
            XO = y'
            you can just take the pseudoinverse
            O = (X.T @ X.T) ** -1 @ y'
    """
    def fit_newt(self):
        self.theta = np.linalg.solve(self.X.T @ self.X, self.X.T @ self.y)


    """
        We can find the SVD of 'X' and then find the pseudoinverse of that, 
        which happens to be:
                X = U @ S @ V.T
                X-1 = V @ S-1 @ U.T (This is because U and V are orthogonal)

        Here 'S' can be taken as a square matric (economy representation).
        This makes U not a square matrix. 
        This is the same thing as the Closed form, but here we don't have to
        calculate the pseudoinverse of X explicitly.

    """
    def fit_svd(self):
        U, S, Vt = np.linalg.svd(self.X, full_matrices=False)
        self.theta = Vt.T @ np.linalg.solve(np.diag(S), U.T @ self.y)


    def predict(self, x: np.ndarray) -> float:
        return self.theta.T @ x
