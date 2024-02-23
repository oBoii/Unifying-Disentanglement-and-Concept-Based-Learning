import numpy as np
import os

from data_module import MNISTDataModule, DSPRITEDataModule, AnimalDataModule
from datasettype import DatasetType
import matplotlib.pyplot as plt


class Utility:
    @staticmethod
    def convert_to_display(samples):
        cnt, height, width = int(np.floor(np.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
        samples = np.transpose(samples, axes=[1, 0, 2, 3])
        # if 1 channel:
        # samples = np.reshape(samples, [height, cnt, cnt, width])

        # if 3 channels (rgb):
        samples = np.reshape(samples, [height, cnt, cnt, width, 3])
        samples = np.transpose(samples, axes=[1, 0, 2, 3, 4])
        samples = np.reshape(samples, [height * cnt, width * cnt, 3])

        # samples = np.transpose(samples, axes=[1, 0, 2, 3])
        # samples = np.reshape(samples, [height * cnt, width * cnt])
        return samples # shape: (64, 64, 3)

    @staticmethod
    def get_files_in_directory(directory):
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    @staticmethod
    def get_directories_in_directory(directory):
        return [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    @staticmethod
    def get_latest_version():
        # outdated was for tensorboard
        # return sorted([int(f.split("_")[1]) for f in Utility.get_directories_in_directory("./lightning_logs")])[-1]

        # now wandb, return the entire file name
        return sorted(Utility.get_directories_in_directory("./wandb"))[-1]  # eg: "run-20211007_123456-3k4j5l6m"

    @staticmethod
    def setup(dataset, batch_size: int = 32, num_workers: int = 1):
        if dataset == DatasetType.MNIST:
            project_name = "autoencoder_mnist"
            data = MNISTDataModule(batch_size=batch_size, num_workers=num_workers)
            im_shape = (1, 28, 28)
        elif dataset == DatasetType.DSPRITE:
            project_name = "autoencoder_dsprites"
            data = DSPRITEDataModule(batch_size=batch_size, num_workers=num_workers)
            im_shape = (1, 64, 64)
        elif dataset == DatasetType.AWA2:
            project_name = "autoencoder_awa"
            data = AnimalDataModule(batch_size=batch_size, num_workers=num_workers)
            im_shape = (3, 64, 64)
        else:
            raise ValueError("Invalid dataset type")

        return data, im_shape, project_name


    @staticmethod
    def display_embeddings(decoder, embeddings, path):
        samples = decoder(embeddings)  # dimension: (100, 1, 64, 64)
        return Utility.display_images(samples, path)

    @staticmethod
    def display_images(images, path):
        images = images.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()  # (100, 64, 64, 1)
        if path is not None:
            plt.imsave(path, Utility.convert_to_display(images), cmap='Greys_r')
        else:
            plt.imshow(Utility.convert_to_display(images), cmap='Greys_r')
            plt.show()

        return images