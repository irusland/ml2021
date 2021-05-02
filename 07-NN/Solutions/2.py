import torch
def nearest_value(X: torch.Tensor, a: float) -> torch.Tensor:
    Y = abs(X - a)
    result = torch.where(Y == Y.min())
    return X[result].min()