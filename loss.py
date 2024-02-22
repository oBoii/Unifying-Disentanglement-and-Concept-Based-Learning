import torch
from architecture import compute_mmd


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weight_for_0, weight_for_1, beta, z_dim):
        super().__init__()
        self.weight_for_0 = weight_for_0
        self.weight_for_1 = weight_for_1
        self.beta = beta

        self.z_dim = z_dim

    def forward(self, input, target, z):
        weights = torch.where(target == 1, self.weight_for_1, self.weight_for_0)
        if self.beta == 0:
            return torch.mean(weights * (input - target) ** 2)
        else:
            true_samples = torch.randn((200, self.z_dim), requires_grad=False, device=input.device)
            mmd = self.beta * compute_mmd(true_samples, z)
            return torch.mean(weights * (input - target) ** 2) + mmd
