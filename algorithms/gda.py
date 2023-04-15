from . import predicters as pred
import numpy as np

class GaussianDiscriminantAnalysis(pred.Predicters):
    def __init__(self, trainX: np.ndarray, trainY: np.ndarray):
        assert trainX.shape[0] == trainY.shape[0]

        self.X = trainX
        self.y = trainY
        
        self.phi = 0
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None

    def calc_phi(self):
        m = self.X.shape[0]
        ones = np.ones(m).reshape(m, 1)
        self.phi =  (self.y.T @ ones) / m

    def calc_mu_0(self):
        m = self.X.shape[0]

        res = (self.X * (1 - self.y))
        ones = np.ones(m).reshape(m, 1)
        self.mu_0 = (res.T @ ones) / ((1 - self.y).T @ ones)
        
    def calc_mu_1(self):
        m = self.X.shape[0]
        
        res = (self.X * self.y)
        ones = np.ones(m).reshape(m, 1)
        self.mu_1 = (res.T @ ones) / (self.y.T @ ones)

    def calc_sigma(self):
        m = self.X.shape[0]
        W =  (np.repeat(self.mu_0.T, m, axis=0) * (1 - self.y))
        W += (np.repeat(self.mu_1.T, m, axis=0) * self.y)

        Z = self.X - W
        self.sigma = (Z.T @ Z) / m

    def fit(self):
        self.calc_phi()
        self.calc_mu_0()
        self.calc_mu_1()
        self.calc_sigma()

    def predict(self, x: np.ndarray) -> int:
        x = x.reshape(x.shape[0], 1)
        n = self.mu_0.shape[0]
        pi = 3.14

        b_0 = (x - self.mu_0)
        b_1 = (x - self.mu_1)

        fac = 1 / ((2 * pi) ** (n / 2) * np.linalg.det(self.sigma) ** 0.5)

        p_x_y0 = fac * np.exp(-0.5 * b_0.T @ np.linalg.solve(self.sigma, b_0))
        p_x_y1 = fac * np.exp(-0.5 * b_1.T @ np.linalg.solve(self.sigma, b_1))

        return 0 if p_x_y0 * (1 - self.phi) > p_x_y1 * self.phi else 1
    
    def print_params(self):
        assert self.phi != 0
        assert self.mu_0 is not None
        assert self.mu_1 is not None
        assert self.sigma is not None

        print(f"""
Phi: {self.phi}
Mu_0: 
{self.mu_0}
Mu_1: 
{self.mu_1}
Sigma:
{self.sigma}
                """)