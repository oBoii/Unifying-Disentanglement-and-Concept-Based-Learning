import os
from glob import glob

import cv2
import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./tmp/MNIST", batch_size: int = 200, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False, transform=ToTensor())
        self.mnist_train = MNIST(self.data_dir, train=True, transform=ToTensor())

    def train_dataloader(self):
        return self._dataloader(self.mnist_train)

    def test_dataloader(self):
        return self._dataloader(self.mnist_test)

    def _dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
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


class AnimalDataset(Dataset):
    # https://github.com/dfan/awa2-zero-shot-learning/blob/master/AnimalDataset.py
    def __init__(self, transform, data_dir='awa2-dataset/AwA2-data/Animals_with_Attributes2_resized'):
        self.transform = transform

        class_to_index = dict()
        # Build dictionary of indices to classes
        with open(f'{data_dir}/classes.txt') as f:
            index = 0
            for line in f:
                class_name = line.split('\t')[1].strip()
                class_to_index[class_name] = index
                index += 1
        self.class_to_index = class_to_index

        img_names = []
        img_index = []
        with open(f'{data_dir}/classes.txt') as f:
            for line in f:
                class_name = line.split('\t')[1].strip()  # split the line and take the second element
                FOLDER_DIR = os.path.join(f'{data_dir}/JPEGImages', class_name)
                file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
                files = glob(file_descriptor)  # glob is used to get all the files in the folder

                class_index = class_to_index[class_name]  # use the class name as the key
                for file_name in files:
                    img_names.append(file_name)
                    img_index.append(class_index)
        self.img_names = img_names
        self.img_index = img_index

    def __getitem__(self, index):
        im = Image.open(self.img_names[index])
        if im.getbands()[0] == 'L':
            im = im.convert('RGB')
        if self.transform:
            im = self.transform(im)
        if im.shape != (3, 64, 64):
            print(f"Image shape is {im.shape} for {self.img_names[index]}")

        im_index = self.img_index[index]

        # im_predicate = self.predicate_binary_mat[im_index, :]
        im_predicate = torch.zeros(85)  # todo
        class_index = self.img_index[index]
        return im, im_predicate
        # im, #im_predicate, self.img_names[index], im_index

    def __len__(self):
        return len(self.img_names)


class AnimalDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = 'awa2-dataset/AwA2-data/Animals_with_Attributes2_resized',
                 batch_size: int = 200, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.awa = AnimalDataset(transform=self.transform, data_dir=self.data_dir)
        # Determine the lengths of splits
        train_len = int(len(self.awa) * 0.7)
        val_len = int(len(self.awa) * 0.15)
        test_len = len(self.awa) - train_len - val_len

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.awa, [train_len, val_len, test_len])

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False)

    def _dataloader(self, dataset, shuffle: bool):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True, shuffle=shuffle)


class DSPRITEDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./dsprites-dataset/", batch_size: int = 100, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.transform = transforms.Compose([
            transforms.ToTensor()])

        # Load the full dataset
        full_dataset = DSPRITEDataset(self.data_dir, transform=self.transform)

        # Determine the lengths of splits
        train_len = int(len(full_dataset) * 0.7)
        val_len = int(len(full_dataset) * 0.15)
        test_len = len(full_dataset) - train_len - val_len

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False)

    def _dataloader(self, dataset, shuffle: bool):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True, shuffle=shuffle)


if __name__ == "__main__":
    # mnist = MNISTDataModule()
    # mnist.setup("train")
    # for x, y in mnist.train_dataloader():
    #     print(x.shape, y.shape)  # torch.Size([200, 1, 28, 28]) torch.Size([200])
    #     break

    awa = AnimalDataset(transform=transforms.Compose([transforms.ToTensor()]))
    print(len(awa))  # 37322
    item = awa[0]
    print(awa[0])
