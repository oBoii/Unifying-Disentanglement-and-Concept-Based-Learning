import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms

from Utility import Utility
from architecture import Encoder, Decoder, compute_mmd


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

                plt.imshow(Utility.convert_to_display(samples), cmap='Greys_r')
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
