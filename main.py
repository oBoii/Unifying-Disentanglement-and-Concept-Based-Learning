import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms


# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 64, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            # Flatten(),
            torch.nn.Flatten(),
            torch.nn.Linear(6272, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, z_dim)
        ])

    def forward(self, x):
        # print("Encoder")
        # print(x.size())
        for layer in self.model:
            x = layer(x)
            # print(x.size())
        return x


class Decoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 7 * 7 * 128),
            torch.nn.ReLU(),
            # Reshape((128, 7, 7,)),
            torch.nn.Unflatten(1, (128, 7, 7)),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
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


def convert_to_display(samples):
    cnt, height, width = int(np.floor(np.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height * cnt, width * cnt])
    return samples


class Model(torch.nn.Module):
    def __init__(self, z_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


def train(
        dataloader: torch.utils.data.DataLoader,
        z_dim=2,
        n_epochs=10,
        use_cuda=True,
        print_every=100,
        plot_every=500
):
    device = torch.device("cuda" if use_cuda else "cpu")
    model: Model = Model(z_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    i = -1
    for epoch in range(n_epochs):
        for images, labels in dataloader:
            i += 1
            optimizer.zero_grad()
            # x = torch.tensor(images, requires_grad=False, dtype=torch.float32).to(device)
            # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
            x = images.clone().detach().requires_grad_(False).to(device)
            true_samples = torch.randn((200, z_dim), requires_grad=False).to(device)

            z, x_reconstructed = model(x)
            mmd = compute_mmd(true_samples, z)
            nll = (x_reconstructed - x).pow(2).mean()  # Negative log likelihood
            loss = nll + mmd
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print(f"Negative log likelihood is {nll.item():.5f}, mmd loss is {mmd.item():.5f}")
            if i % plot_every == 0:
                gen_z = torch.randn((100, z_dim), requires_grad=False).to(device)

                samples = model.decoder(gen_z)
                samples = samples.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()

                plt.imshow(convert_to_display(samples), cmap='Greys_r')
                plt.show()

    return model


if __name__ == "__main__":
    # model = Model(z_dim=2)
    # rnd_input = torch.randn(1, 1, 28, 28)
    # z, x_reconstructed = model(rnd_input)
    # print(z.size(), x_reconstructed.size())

    # Load the MNIST dataset
    batch_size = 200
    mnist_train = torch.utils.data.DataLoader(
        MNIST("./tmp/MNIST", train=True, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
              ])),
        batch_size=batch_size, shuffle=True, num_workers=3,
        pin_memory=True
    )

    z_dim = 2
    model = train(mnist_train, z_dim=z_dim, n_epochs=10)
