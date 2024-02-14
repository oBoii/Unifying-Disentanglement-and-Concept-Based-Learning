import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import numpy as np

from architecture import compute_mmd
from data_module import MNISTDataModule, DSPRITEDataModule
from utility import Utility
# from architecture import Encoder, Decoder, compute_mmd
from architecture_burgess import Encoder, Decoder
import matplotlib.pyplot as plt


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weight_for_0, weight_for_1, beta):
        super().__init__()
        self.weight_for_0 = weight_for_0
        self.weight_for_1 = weight_for_1
        self.beta = beta

    def forward(self, input, target, z):
        weights = torch.where(target == 1, self.weight_for_1, self.weight_for_0)
        if self.beta == 0:
            return torch.mean(weights * (input - target) ** 2)
        else:
            true_samples = torch.randn((200, z_dim), requires_grad=False, device=input.device)
            mmd = self.beta * compute_mmd(true_samples, z)
            return torch.mean(weights * (input - target) ** 2) + mmd


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func: WeightedMSELoss = WeightedMSELoss(1.0, 10.0, 0)

        self.test_losses = []

    def training_step(self, batch, batch_idx):
        x, y = batch  # x.shape: (200, 1, 64, 64), y.shape: (200, 1, 6)
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        loss = self.loss_func(x_reconstructed, x, z)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        loss = self.loss_func(x_reconstructed, x, z)
        self.test_losses.append(loss)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        self.log("avg_test_loss", avg_loss)
        self.test_losses = []  # reset for the next epoch


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

    is_MNIST: bool = False
    z_dim = 32
    height = 28 if is_MNIST else 64
    encoder = Encoder(latent_dim=z_dim, img_size=(1, height, height))
    decoder = Decoder(latent_dim=z_dim, img_size=(1, height, height))

    autoencoder = LitAutoEncoder(encoder, decoder)

    data = MNISTDataModule() if is_MNIST else DSPRITEDataModule(workers=10)

    # Train the model
    trainer = L.Trainer(limit_train_batches=.1, max_epochs=5, accelerator="gpu", devices="1",
                        callbacks=[CustomCallbacks(plot_ever_n_epoch=4)])
    trainer.fit(model=autoencoder, datamodule=data)
    trainer.test(model=autoencoder, datamodule=data)

    # tensorboard --logdir .
    # tensorboard --logdir ./lightning_logs
