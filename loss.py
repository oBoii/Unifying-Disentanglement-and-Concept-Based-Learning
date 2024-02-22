import torch
from architecture import compute_mmd


class MSE(torch.nn.Module):
    def __init__(self, beta, z_dim):
        super().__init__()
        self.beta = beta
        self.z_dim = z_dim

    def forward(self, input, target, z):
        if self.beta == 0:
            return torch.mean((input - target) ** 2)
        else:
            true_samples = torch.randn((200, self.z_dim), requires_grad=False, device=input.device)
            mmd = self.beta * compute_mmd(true_samples, z)
            return torch.mean((input - target) ** 2) + mmd