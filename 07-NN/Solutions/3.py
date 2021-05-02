from torch import log as ln
from torch import arctan, arcsin

def f(x):
    return ln(arctan(x)) / (arcsin(x))**2

def derivative(X: torch.Tensor) -> torch.Tensor:
    X = X.requires_grad_(True)
    y = f(X)
    y.backward(gradient=torch.ones_like(y))
    return X.grad