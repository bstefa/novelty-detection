import os
import torch
import numpy as np

from typing import Optional
from torchvision import datasets, transforms

from .base_datamodule import BaseDataModule


class EMNISTDataModule(BaseDataModule):

    def __init__(
            self,
            root_data_path: str,
            batch_size: int = 8,
            train_fraction: float = 0.8,
            download: bool = False,
            data_transforms: Optional[transforms.Compose] = None,
            **kwargs):
        super(EMNISTDataModule, self).__init__()
        print(f'[INFO] Loading {self.__class__.__name__}...')

        self._download = download
        self._root_data_path = root_data_path
        self._batch_size = batch_size if batch_size is not None else 8
        self._train_fraction = train_fraction if train_fraction is not None else 0.8
        self._data_transforms = data_transforms if data_transforms is not None else transforms.ToTensor()

        if self._download:
            if not os.path.isdir(self._root_data_path):
                print("Creating download directory...")
                os.makedirs(self._root_data_path)

    def prepare_data(self):

        if self._download:
            datasets.EMNIST(self._root_data_path, split='mnist', train=True, download=True)
            datasets.EMNIST(self._root_data_path, split='mnist', train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit' or stage == 'train' or stage is None:
            trainval_set = datasets.EMNIST(
                self._root_data_path,
                split='mnist',
                train=True,
                transform=self._data_transforms)

            train_size = int(np.floor(len(trainval_set) * self._train_fraction))
            val_size = len(trainval_set) - train_size
            # Since setup is called from every process, setting state here is okay
            self._train_set, self._val_set = torch.utils.data.random_split(trainval_set, [train_size, val_size])

        elif stage == 'test' or stage is None:
            self._test_set = datasets.EMNIST(
                self._root_data_path,
                split='mnist',
                train=False,
                transform=self._data_transforms)

        else:
            raise ValueError('Only accepts \'train\', \'test\', or None for stage.')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_set,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_set,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=8)
