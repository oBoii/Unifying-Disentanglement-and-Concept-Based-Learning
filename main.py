import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import numpy as np
from utility import Utility
from architecture import Encoder, Decoder, compute_mmd
import matplotlib.pyplot as plt

from data_module import MNISTDataModule, DSPRITEDataModule


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch  # x.shape: (200, 1, 64, 64), y.shape: (200, 1, 6)
        true_samples = torch.randn((200, z_dim), requires_grad=False, device=x.device)

        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        # mmd = compute_mmd(true_samples, z)
        nll = (x_reconstructed - x).pow(2).mean()  # Negative log likelihood
        loss = nll #+ mmd

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
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
            min, max = np.min(samples), np.max(samples)
            plt.imshow(Utility.convert_to_display(samples), cmap='Greys_r')
            plt.show()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    is_MNIST: bool = False
    z_dim = 32
    height = 28 if is_MNIST else 64
    encoder = Encoder(z_dim, height)
    decoder = Decoder(z_dim, height)

    autoencoder = LitAutoEncoder(encoder, decoder)

    data = MNISTDataModule() if is_MNIST else DSPRITEDataModule(workers=10)

    # Train the model
    trainer = L.Trainer(limit_train_batches=.1, max_epochs=10, accelerator="gpu", devices="1",
                        callbacks=[CustomCallbacks(plot_ever_n_epoch=4)])
    trainer.fit(model=autoencoder, datamodule=data)

    # tensorboard --logdir .
    # tensorboard --logdir ./lightning_logs
