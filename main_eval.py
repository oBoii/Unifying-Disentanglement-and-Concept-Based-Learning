import torch
import lightning as L
from architecture import Encoder, Decoder
# from architecture_burgess import Encoder, Decoder
from data_module import DSPRITEDataModule
from main import LitAutoEncoder
import numpy as np
import matplotlib.pyplot as plt

from utility import Utility
import yaml


def display_embeddings(embeddings, path):
    samples = decoder(embeddings)  # dimension: (100, 1, 64, 64)
    display_images(samples, path)


def display_images(images, path):
    images = images.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()  # (100, 64, 64, 1)
    if path is not None:
        plt.imsave(path, Utility.convert_to_display(images), cmap='Greys_r')
    else:
        plt.imshow(Utility.convert_to_display(images), cmap='Greys_r')
        plt.show()


if __name__ == "__main__":
    # Load checkpoint
    z_dim = 2  # 32  # 4096
    height = 64
    encoder = Encoder(latent_dim=z_dim, img_size=(1, height, height))
    decoder = Decoder(latent_dim=z_dim, img_size=(1, height, height))

    version_name = Utility.get_latest_version()

    logs_dir = f"./wandb/{version_name}/files"
    checkpoint = f"{logs_dir}/checkpoints/last.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

    # Choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

    datamodule = DSPRITEDataModule(workers=1)
    datamodule.prepare_data()
    datamodule.setup("test")
    for x, y in datamodule.test_dataloader():
        x = x.to(autoencoder.device)
        x = x[:100]  # dimension: (100, 1, 64, 64)
        # save in the checkpoint directory
        display_images(x, path=f"{logs_dir}/eval_original_images.png")
        z = encoder(x)
        display_embeddings(z, path=f"{logs_dir}/eval_im_enc_decode.png")
        break

    datamodule.teardown("test")

    # gen_z = torch.randn((100, z_dim), requires_grad=False, device=autoencoder.device)
    # display_embeddings(gen_z)

# interesting models are: 104 (32 dim)
# 105 (8 dim), 106 (2 dim)
# 103 ? 4096 dim
