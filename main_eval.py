import torch
import lightning as L
# from architecture import Encoder, Decoder
from architecture_burgess import Encoder, Decoder
from data_module import DSPRITEDataModule
from main import LitAutoEncoder
import numpy as np
import matplotlib.pyplot as plt

from utility import Utility


def display_embeddings(embeddings):
    samples = decoder(embeddings)  # dimension: (100, 1, 64, 64)
    display_images(samples)


def display_images(images):
    images = images.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()  # (100, 64, 64, 1)
    plt.imshow(Utility.convert_to_display(images), cmap='Greys_r')
    plt.show()


if __name__ == "__main__":

    # Load checkpoint
    z_dim = 32
    height = 64
    encoder = Encoder(latent_dim=z_dim, img_size=(1, height, height))
    decoder = Decoder(latent_dim=z_dim, img_size=(1, height, height))
    version = "97"  # 49
    checkpoint_dir = f"./lightning_logs/version_{version}/checkpoints"
    files_in_checkpoint_dir = Utility.get_files_in_directory(checkpoint_dir)
    checkpoint = f"{checkpoint_dir}/{files_in_checkpoint_dir[-1]}"
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
        display_images(x)
        z = encoder(x)
        display_embeddings(z)
        break

    datamodule.teardown("test")

    # gen_z = torch.randn((100, z_dim), requires_grad=False, device=autoencoder.device)
    # display_embeddings(gen_z)
