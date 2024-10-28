import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch


def mmd(x, y):
    Kxx = pairwise_distances(x, x)
    Kyy = pairwise_distances(y, y)
    Kxy = pairwise_distances(x, y)

    m = x.shape[0]
    n = y.shape[0]
    
    c1 = 1 / ( m * (m - 1))
    A = np.sum(Kxx - np.diag(np.diagonal(Kxx)))

    # Term II
    c2 = 1 / (n * (n - 1))
    B = np.sum(Kyy - np.diag(np.diagonal(Kyy)))

    # Term III
    c3 = 1 / (m * n)
    C = np.sum(Kxy)

    # estimate MMD
    mmd_est = -0.5*c1*A - 0.5*c2*B + c3*C
    
    return mmd_est


class TensorSampler:
    def __init__(self, tensor, device='cuda'):
        self.device = device
        self.tensor = torch.clone(tensor).to(device)
        
    def sample(self, size=5):
        assert size <= self.tensor.shape[0]
        
        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)
