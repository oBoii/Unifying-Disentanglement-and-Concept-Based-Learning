import torch
import lightning as L
from architecture import Encoder, Decoder
from datasettype import DatasetType
from main import LitAutoEncoder
import numpy as np
import matplotlib.pyplot as plt
from utility import Utility
from args import parse_args


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
    args = parse_args()
    z_dim = args.z
    dataset = DatasetType(args.dataset)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    limit_train_batches = args.limit_train_batches
    run_name = args.run_name

    data_module, im_shape, project_name = Utility.setup(dataset)

    encoder = Encoder(latent_dim=z_dim, img_size=im_shape)
    decoder = Decoder(latent_dim=z_dim, img_size=im_shape)

    if run_name == "":
        run_name = Utility.get_latest_version()
        print(f"No run name provided. Using the latest version. Run name: {run_name}")

    logs_dir = f"./wandb/{run_name}/files"
    checkpoint = f"{logs_dir}/checkpoints/last.ckpt"
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder, z_dim=z_dim, lr=lr)

    # Choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

    data_module.prepare_data()
    data_module.setup("test")
    for x, y in data_module.test_dataloader():
        x = x.to(autoencoder.device)
        x = x[:100]  # dimension: (100, 1, 64, 64)
        # save in the checkpoint directory
        display_images(x, path=f"{logs_dir}/eval_original_images.png")
        z = encoder(x)
        display_embeddings(z, path=f"{logs_dir}/eval_im_enc_decode.png")
        break

    data_module.teardown("test")

    # gen_z = torch.randn((100, z_dim), requires_grad=False, device=autoencoder.device)
    # display_embeddings(gen_z)

# interesting models are: 104 (32 dim)
# 105 (8 dim), 106 (2 dim)
# 103 ? 4096 dim
