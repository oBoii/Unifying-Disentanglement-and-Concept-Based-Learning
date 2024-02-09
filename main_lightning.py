import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import numpy as np

from Utility import Utility
from architecture import Encoder, Decoder, compute_mmd
import matplotlib.pyplot as plt


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        true_samples = torch.randn((200, z_dim), requires_grad=False).to(x.device)
        # x = x.view(x.size(0), -1)

        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        mmd = compute_mmd(true_samples, z)
        nll = (x_reconstructed - x).pow(2).mean()  # Negative log likelihood
        loss = nll + mmd

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CustomCallbacks(L.Callback):
    def __init__(self, plot_ever_n_epoch):
        super().__init__()
        self.plot_ever_n_epoch = plot_ever_n_epoch

    def on_train_epoch_end(self, trainer, pl_module: LitAutoEncoder):
        # Every 10th epoch, generate some images
        if trainer.current_epoch % self.plot_ever_n_epoch == 0:
            gen_z = torch.randn((100, z_dim), requires_grad=False, device=pl_module.device)

            samples = pl_module.decoder(gen_z)
            samples = samples.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
            plt.imshow(Utility.convert_to_display(samples), cmap='Greys_r')
            plt.show()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    z_dim = 2
    encoder = Encoder(z_dim)
    decoder = Decoder(z_dim)

    autoencoder = LitAutoEncoder(encoder, decoder)
    dataset = MNIST("./tmp/MNIST", download=True, transform=ToTensor())
    # pin_memory=True is used to speed up the data transfer from CPU to GPU
    # persistent_workers=True is used to speed up the data loading process
    train_loader = utils.data.DataLoader(
        dataset, batch_size=200, num_workers=4, shuffle=True, pin_memory=True,
        persistent_workers=True)

    # Train the model
    trainer = L.Trainer(limit_train_batches=1.0, max_epochs=10, accelerator="gpu", devices="1",
                        callbacks=[CustomCallbacks(plot_ever_n_epoch=4)])
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # Load checkpoint
    checkpoint = "./lightning_logs/version_6/checkpoints/epoch=0-step=100.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # Choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

    # Embed 4 fake images!
    fake_image_batch = torch.rand((4, 1, 28, 28), device=autoencoder.device)
    embeddings = encoder(fake_image_batch)
    print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

    # tensorboard --logdir .
