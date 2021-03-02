import os
import pytorch_lightning as pl
from typing import Optional, Generic
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

class MNISTDataModule(pl.LightningDataModule):

    def __init__(
            self,
            config: dict,
            transforms: Optional[transforms.Compose] = None
        ):

        super(MNISTDataModule, self).__init__()

        self.download_dir = config['dataset_download_dir']
        self.batch_size = config['batch_size']
        self.transforms = transforms

        if os.path.isdir(self.download_dir) == False:
            print("Creating download directory")
            os.makedirs(self.download_dir)

    def prepare_data(self):

        datasets.MNIST(self.download_dir,
            train=True,
            download=True)

        datasets.MNIST(self.download_dir,
            train=False,
            download=True)

    def setup(self, stage: Optional[str]=None):

        data = datasets.MNIST(self.download_dir,
                              train=True,
                              transform=self.transforms)
        
        self.train_data, self.val_data = random_split(data, [55000, 5000])
  
        self.test_data = datasets.MNIST(self.download_dir, 
                                        train=False, 
                                        transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=4)

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=4)

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=4)