import lightning as L
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

from architecture import Encoder


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./tmp/MNIST", batch_size: int = 200):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False, transform=ToTensor())
        self.mnist_train = MNIST(self.data_dir, train=True, transform=ToTensor())

    def train_dataloader(self):
        return self._dataloader(self.mnist_train)

    def test_dataloader(self):
        return self._dataloader(self.mnist_test)

    def _dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=19, pin_memory=True,
                          persistent_workers=True, shuffle=True)


class DspriteDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_folder = ImageFolder(data_folder, transform=transform)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        return self.image_folder[idx]

class DSPRITEDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./dsprites-dataset/", batch_size: int = 200, subset: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.subset = subset

    def setup(self, stage: str):
        file_name = self.data_dir + ("dsprites_subset.npz" if self.subset else "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        dataset = np.load(file_name, encoding='bytes')
        images = dataset['imgs']
        images = np.expand_dims(images, axis=1)
        images = torch.tensor(images, dtype=torch.float32)
        images = images / 255.0  # shape: (737280, 1, 64, 64)

        latent_values = dataset['latents_values']
        latent_values = np.expand_dims(latent_values, axis=1)
        latents_values = torch.tensor(latent_values, dtype=torch.float32)

        latents_classes = dataset['latents_classes']

        data = TensorDataset(images, latents_values)

        # Split the dataset into training and test
        n = len(data)
        n_train = int(0.8 * n)
        n_test = n - n_train
        self.dsprite_train, self.dsprite_test = random_split(data, [n_train, n_test])

    def train_dataloader(self):
        return self._dataloader(self.dsprite_train)

    def test_dataloader(self):
        return self._dataloader(self.dsprite_test)

    def _dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=19, pin_memory=True,
                          persistent_workers=True, shuffle=True)


if __name__ == "__main__":
    mnist = MNISTDataModule()
    mnist.setup("train")
    for x, y in mnist.train_dataloader():
        print(x.shape, y.shape) # torch.Size([200, 1, 28, 28]) torch.Size([200])
        break