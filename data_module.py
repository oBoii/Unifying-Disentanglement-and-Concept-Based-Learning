import lightning as L
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import os
import cv2

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


class DSPRITEDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_folder = f"{data_folder}/input"
        self.latent_classes_folder = f"{data_folder}/latent_classes"
        self.latent_values_folder = f"{data_folder}/latent_values"
        self.transform = transform

        # count the number of files in the folder
        self.nb_files = 737_280
        # len(
        # [name for name in os.listdir(self.image_folder) if os.path.isfile(os.path.join(self.image_folder, name))])

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        img_path = f"{self.image_folder}/{idx}.png"
        latent_classes_path = f"{self.latent_classes_folder}/{idx}.npy"
        latent_values_path = f"{self.latent_values_folder}/{idx}.npy"

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = self.transform(image)
        label = np.load(latent_classes_path)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


class DSPRITEDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./dsprites-dataset/", batch_size: int = 100, workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage: str):
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def train_dataloader(self):
        train_dataset = DSPRITEDataset(self.data_dir, transform=self.transform)
        return self._dataloader(train_dataset, shuffle=True)

    def test_dataloader(self):
        # warning: currently, the test dataset is the same as the train dataset
        test_dataset = DSPRITEDataset(self.data_dir, transform=self.transform)
        return self._dataloader(test_dataset, shuffle=False)

    def _dataloader(self, dataset, shuffle: bool):
        # return DataLoader(dataset, batch_size=self.batch_size, num_workers=19, pin_memory=True,
        #                   persistent_workers=True, shuffle=True)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True,
                          persistent_workers=True, shuffle=shuffle)


if __name__ == "__main__":
    mnist = MNISTDataModule()
    mnist.setup("train")
    for x, y in mnist.train_dataloader():
        print(x.shape, y.shape)  # torch.Size([200, 1, 28, 28]) torch.Size([200])
        break
