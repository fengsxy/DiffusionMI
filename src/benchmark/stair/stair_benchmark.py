import numpy as np
import torch
from typing import Tuple


class CorrelatedGaussianTask:
    def __init__(self, name: str, dim: int, rho: float, cubic: bool = False):
        self.name = name
        self.dim_x = dim
        self.dim_y = dim
        self.rho = rho
        self.cubic = cubic
        self.mutual_information = self.compute_mutual_information()

    def compute_mutual_information(self) -> float:
        return -0.5 * np.log(1 - self.rho**2) * self.dim_x

    def sample_correlated_gaussian(self,rho=0.5, dim=20, batch_size=128, cubic=None):
        """Generate samples from a correlated Gaussian distribution."""
        
        x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
        y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps
        if cubic is not None:
            y = y ** 3
        return x, y

    def sample(self, n: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            torch.manual_seed(seed)
        x, y = self.sample_correlated_gaussian( rho=0.5,dim=self.dim_x, batch_size=n, cubic=self.cubic)
        return x.numpy(), y.numpy()
    
if __name__ == "__main__":
    task = CorrelatedGaussianTask("correlated_gaussian", 20, 0.5)
    x, y = task.sample(128)
    print(x.shape, y.shape)
    print(task.mutual_information)