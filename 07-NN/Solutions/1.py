from torch import cos, sin
def cannabola(theta: torch.Tensor) -> torch.Tensor:
    rho = (1 + 0.9*cos(8*theta)) * (1 + 0.1*cos(24*theta))*(0.9 + 0.05*cos(200*theta))*(1 + sin(theta))
    x =   rho*cos(theta)
    y =   rho*sin(theta)
    return rho, x, y