# pytorch lightning validation printing bug:
# https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning/66731318#66731318

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from architecture import Encoder, Decoder
from args import parse_args
from datasettype import DatasetType
from lit_autoencoder import LitAutoEncoder
from utility import Utility


class CustomCallbacks(L.Callback):
    def __init__(self, plot_ever_n_epoch, z_dim):
        super().__init__()
        self.plot_ever_n_epoch = plot_ever_n_epoch
        self.z_dim = z_dim

    def on_train_epoch_end(self, trainer, pl_module: LitAutoEncoder):
        # Every 10th epoch, generate some images
        if trainer.current_epoch % self.plot_ever_n_epoch == 0:
            gen_z = torch.randn((100, self.z_dim), requires_grad=False, device=pl_module.device)

            samples = pl_module.decoder(gen_z)
            samples = samples.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
            plt.imshow(Utility.convert_to_display(samples), cmap='Greys_r')

            # store in log folder (wandb log folder)
            wandb_folder = wandb.run.dir
            plt.savefig(f"{wandb_folder}/epoch_{trainer.current_epoch}.png")


def main():
    args = parse_args()
    z_dim = args.z
    dataset = DatasetType(args.dataset)
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    limit_train_batches = args.limit_train_batches

    torch.set_float32_matmul_precision('medium')

    data, im_shape, project_name = Utility.setup(dataset, batch_size, num_workers)

    wandb.init(project="autoencoder_awa")
    wandb_logger = WandbLogger()
    checkpoint_dir = f"{wandb.run.dir}/checkpoints"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, verbose=True, monitor='val_loss',
                                          mode='min', save_last=True)

    encoder = Encoder(latent_dim=z_dim, img_size=im_shape)
    decoder = Decoder(latent_dim=z_dim, img_size=im_shape)

    autoencoder = LitAutoEncoder(encoder, decoder, z_dim, lr)

    # Train the model
    trainer = L.Trainer(limit_train_batches=limit_train_batches, max_epochs=epochs, accelerator="gpu", devices="1",
                        logger=wandb_logger,
                        callbacks=[CustomCallbacks(plot_ever_n_epoch=4, z_dim=z_dim), checkpoint_callback])
    trainer.fit(model=autoencoder, datamodule=data)
    trainer.test(model=autoencoder, datamodule=data)

    # second to last element is the run name
    # if windows:
    if "\\" in wandb.run.dir:
        run_name = wandb.run.dir.split("\\")[-2]
    else:  # if linux
        run_name = wandb.run.dir.split("/")[-2]

    wandb.finish()

    print(f"Run name: {run_name}")
    return run_name


# enum
if __name__ == "__main__":
    main()

    # tensorboard --logdir .
    # tensorboard --logdir ./lightning_logs

    # wandb commands:
    # wandb login (or wandb login your-api-key)
    # wandb online
