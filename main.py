# pytorch lightning validation printing bug:
# https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning/66731318#66731318

import torch
from torch import optim
import lightning as L

from data_module import MNISTDataModule, DSPRITEDataModule, AnimalDataModule
from datasettype import DatasetType
from utility import Utility
from architecture import Encoder, Decoder
import matplotlib.pyplot as plt
from loss import MSE

import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from args import parse_args


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder, z_dim: int, lr: float, beta: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_func: MSE = MSE(beta, z_dim)
        self.lr = lr

        self.test_losses = []

        self.save_hyperparameters(ignore=["encoder", "decoder", "loss_func"])

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

            # store in log folder (wandb log folder)
            wandb_folder = wandb.run.dir
            plt.savefig(f"{wandb_folder}/epoch_{trainer.current_epoch}.png")


# enum
if __name__ == "__main__":
    args = parse_args()
    z_dim = args.z
    dataset = DatasetType(args.dataset)
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    limit_train_batches = args.limit_train_batches
    assert args.run_name == "", "Run name only useful for evaluation"

    torch.set_float32_matmul_precision('medium')

    data, im_shape, project_name = Utility.setup(dataset)

    wandb.init(project="autoencoder_awa")
    wandb_logger = WandbLogger()
    checkpoint_dir = f"{wandb.run.dir}/checkpoints"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, verbose=True, monitor='val_loss',
                                          mode='min', save_last=True)

    encoder = Encoder(latent_dim=z_dim, img_size=im_shape)
    decoder = Decoder(latent_dim=z_dim, img_size=im_shape)

    autoencoder = LitAutoEncoder(encoder, decoder, z_dim, lr)

    # Train the model
    trainer = L.Trainer(limit_train_batches=.1, max_epochs=epochs, accelerator="gpu", devices="1",
                        logger=wandb_logger,
                        callbacks=[CustomCallbacks(plot_ever_n_epoch=4), checkpoint_callback])
    trainer.fit(model=autoencoder, datamodule=data)
    trainer.test(model=autoencoder, datamodule=data)

    wandb.finish()

    # tensorboard --logdir .
    # tensorboard --logdir ./lightning_logs

    # wandb commands:
    # wandb login (or wandb login your-api-key)
    # wandb online
