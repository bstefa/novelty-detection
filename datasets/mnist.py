import os
import pytorch_lightning as pl
from typing import Optional, Generic
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from pprint import pprint as pp

class MNISTDataModule(pl.LightningDataModule):

    def __init__(
            self,
            config: dict,
            data_transforms: Optional[transforms.Compose]=None
        ):

        super(MNISTDataModule, self).__init__()

        self.download_dir = config['dataset_download_dir']
        self.batch_size = config['batch_size']

        if data_transforms is not None:
            self.data_transforms = data_transforms
        else:
            self.data_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

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
                              transform=self.data_transforms)
        
        self.train_data, self.val_data = random_split(data, [55000, 5000])
  
        self.test_data = datasets.MNIST(self.download_dir, 
                                        train=False, 
                                        transform=self.data_transforms)

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

    @property
    def len_train_data(self):
        return len(self.train_data)

    @property
    def len_val_data(self):
        return len(self.val_data)

    @property
    def len_test_data(self):
        return len(self.test_data)
    

if __name__ == '__main__':

    # Load test configs
    test_configs = {
                    'dataset_download_dir': '/home/fenrir/Documents/novelty-detection/downloads',
                    'batch_size': 32
                    }

    # Initialize transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create data module object
    datamodule = MNISTDataModule(test_configs, data_transforms)

    # Prepare and setup data
    datamodule.prepare_data()
    datamodule.setup()

    # Load train_dataloader
    dataloader = datamodule.train_dataloader()

    # Grabe a pair of (batch, x) and (batch, y)
    sample, labels = next(iter(dataloader))

    pp(sample.shape)
    pp(labels.shape)