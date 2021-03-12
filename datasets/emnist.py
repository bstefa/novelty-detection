import os
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Optional, Generic, Union

from torchvision import datasets, transforms
from pprint import pprint as pp

class EMNISTDataModule(pl.LightningDataModule):

    def __init__(
            self,
            root_data_path: str,
            batch_size: int=8,
            train_fraction: float=0.8,
            download: bool=False,
            data_transforms: Optional[transforms.Compose]=None
        ):
        super(EMNISTDataModule, self).__init__()

        self._download = download
        self._root_data_path = root_data_path
        self._batch_size = batch_size
        self._train_fraction = train_fraction
        self._data_transforms = data_transforms if data_transforms is not None else transforms.ToTensor()

        if self._download:
            if os.path.isdir(self._root_data_path) == False:
                print("Creating download directory...")
                os.makedirs(self._root_data_path)

    def prepare_data(self):

        if self._download:
            datasets.EMNIST(
                self._root_data_path,
                split='mnist',
                train=True,
                download=True)

            datasets.EMNIST(
                self._root_data_path,
                split='mnist',
                train=False,
                download=True)

    def setup(self, stage: Optional[str]=None):

        if stage == 'fit' or stage == 'train' or stage is None:
            trainval_set = datasets.EMNIST(
                self._root_data_path,
                split='digits',
                train=True,
                transform=self._data_transforms
            )
            # Since setup is called from every process, setting state here is okay
            train_size = int(np.floor(len(trainval_set) * self._train_fraction))
            val_size = len(trainval_set) - train_size

            self._train_set, self._val_set = torch.utils.data.random_split(trainval_set, [train_size, val_size])

        elif stage == 'test' or stage is None:
            self._test_set = datasets.EMNIST(
                self._root_data_path,
                split='digits',
                train=False,
                transform=self._data_transforms
            )

        else:
            raise ValueError('Only accepts \'train\', \'test\', or None for stage.')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_set,
            batch_size=self._batch_size,
            num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test_set,
            batch_size=self._batch_size,
            num_workers=8)

    @property
    def train_size(self):
        assert hasattr(self, '_train_set'), 'Need to setup data before getting dataset length'
        return len(self._train_set)

    @property
    def val_size(self):
        return len(self._val_set)

    @property
    def test_size(self):
        return len(self._test_set)

    @property
    def batch_size(self):
        return self._batch_size
    

if __name__ == '__main__':

    # Load test configs
    test_configs = {
                    'root_data_path': '/home/brahste/Datasets/toy',
                    'download': True
                    }

    # Create data module object
    datamodule = EMNISTDataModule(**test_configs)

    # Prepare and setup data
    datamodule.prepare_data()
    datamodule.setup('train')

    # Load train_dataloader
    dataloader = datamodule.train_dataloader()
    print(datamodule.train_size)

    # Grabe a pair of (batch, x) and (batch, y)
    sample, labels = next(iter(dataloader))

    pp(sample.shape)
    pp(labels.shape)