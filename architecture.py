import torch


# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, img_size: tuple, latent_dim: int = 10):
        channels, height, width = img_size
        z_dim = latent_dim

        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(channels, 64, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * (height // 4) * (height // 4), 1024),  # if h=64 -> 32768
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, z_dim)
        ])

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, img_size: tuple, latent_dim: int = 10):
        channels, height, width = img_size
        z_dim = latent_dim

        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 128 * (height // 4) * (height // 4)),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (128, height // 4, height // 4)),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, channels, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / torch.tensor(dim, dtype=torch.float32)

    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd
