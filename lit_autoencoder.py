from typing import Optional

import lightning as L
from architecture import Encoder, Decoder
from loss import MSE
import torch
from torch import optim


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder, z_dim: int, lr: float, beta: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func: MSE = MSE(beta, z_dim)
        self.lr = lr

        self.test_losses = []

        self.save_hyperparameters(ignore=["encoder", "decoder", "loss_func", "wandb"])
        # if wandb:
        #     wandb.config.update()

    def training_step(self, batch, batch_idx):
        x, y = batch  # x.shape: (200, 1, 64, 64), y.shape: (200, 1, 6)
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)

        loss = self.loss_func(x_reconstructed, x, z)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    # validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        loss = self.loss_func(x_reconstructed, x, z)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
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
