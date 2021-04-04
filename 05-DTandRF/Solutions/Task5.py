import numpy as np
from scipy.stats import sem


def get_stats(X: np.array, B: int) -> tuple:
    M = 10
    mean = 0

    SE = 0
    for _ in range(B):
        X_new = get_bootstrap_samples(X, B)
        mean += X_new.mean()
        SE += sem(X_new)

    return mean / B, SE / B


def get_bootstrap_samples(X, B):
    return np.random.choice(X, size=len(X))
