# pytorch lightning validation printing bug:
# https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning/66731318#66731318
from argparse import Namespace

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
    def __init__(self, plot_ever_n_epoch, z_dim, wandb_logger: WandbLogger):
        super().__init__()
        self.plot_ever_n_epoch = plot_ever_n_epoch
        self.z_dim = z_dim
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: LitAutoEncoder):
        # Every 10th epoch, generate some images
        if trainer.current_epoch % self.plot_ever_n_epoch == 0:
            gen_z = torch.randn((100, self.z_dim), requires_grad=False, device=pl_module.device)

            samples = pl_module.decoder(gen_z)
            samples = samples.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
            im = Utility.convert_to_display(samples)
            plt.imshow(im, cmap='Greys_r')

            # store in log folder (wandb log folder)
            wandb_folder = wandb.run.dir
            plt.savefig(f"{wandb_folder}/epoch_{trainer.current_epoch}.png")

            # log to wandb
            self.wandb_logger.log_image("samples_gaussian", images=[im], step=trainer.global_step)

    # when training is done
    def on_train_end(self, trainer, pl_module):
        # encode and decode some images
        data_module, im_shape, project_name = Utility.setup(DatasetType.AWA2, batch_size=200, num_workers=1)
        data_module.prepare_data()
        data_module.setup("test")
        for x, y in data_module.test_dataloader():
            x = x.to(pl_module.device)
            x = x[:100]
            z = pl_module.encoder(x)
            x_reconstructed = pl_module.decoder(z)

            x = x.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()
            ims_gt = Utility.convert_to_display(x)
            x_reconstructed = x_reconstructed.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()

            ims_enc_dec = Utility.convert_to_display(x_reconstructed)

            # log to wandb
            self.wandb_logger.log_image("samples_gt_vs_encDec", images=[ims_gt, ims_enc_dec])
            break


def main():
    args: Namespace = parse_args()
    z_dim = args.z
    dataset = DatasetType(args.dataset)
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    limit_train_batches = args.limit_train_batches

    torch.set_float32_matmul_precision('medium')

    data, im_shape, project_name = Utility.setup(dataset, batch_size, num_workers)

    wandb.init(project="autoencoder_awa",
               tags=[f"z={z_dim}", f"lr={lr}", f"batch_size={batch_size}"],
                name=f"z={z_dim}_lr={lr}_batch_size={batch_size}")

    wandb_logger = WandbLogger()

    # args to dict
    # Log hyperparameters before training
    # wandb.config.update(vars(args))

    # for key, value in vars(args).items():
    #     wandb.log({f"config/{key}": value})

    checkpoint_dir = f"{wandb.run.dir}/checkpoints"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, verbose=True, monitor='val_loss',
                                          mode='min', save_last=True)

    encoder = Encoder(latent_dim=z_dim, img_size=im_shape)
    decoder = Decoder(latent_dim=z_dim, img_size=im_shape)

    autoencoder = LitAutoEncoder(encoder, decoder, z_dim, lr)

    # Train the model
    trainer = L.Trainer(limit_train_batches=limit_train_batches, max_epochs=epochs, accelerator="gpu", devices="1",
                        logger=wandb_logger,
                        callbacks=[CustomCallbacks(plot_ever_n_epoch=4, z_dim=z_dim, wandb_logger=wandb_logger),
                                   checkpoint_callback])
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
