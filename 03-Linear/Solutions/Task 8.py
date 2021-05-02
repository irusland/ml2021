import numpy as np

def log_gradient_step(w: np.array, X: np.array, y: np.array)-> np.array:
    g = X.T @ (sigma(X @ w) - y)
    return g

def sigma(z):
    return 1.0 / (1 + np.exp(-z))