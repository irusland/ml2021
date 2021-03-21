import numpy as np

def gradient_step_l2(w: np.array, X: np.array, y: np.array, lamb: float) -> np.array:
    # Градиент для регуляризации.
    # ∇wL=X.T(Xw−y)
    without_coef = np.copy(w)
    without_coef[-1] = 0
    gradient = np.dot(X.T, (np.dot(X,w) - y)) 
    return gradient + lamb * without_coef