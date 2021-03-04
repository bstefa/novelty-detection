# Datasets used in Novelty detection experiments
# Author: Braden Stefanuk
# Created: Dec 17, 2020
from abc import ABC

import torch
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from torchvision import transforms
from skimage import io, transform
from sklearn.model_selection import train_test_split
from utils import tools
from typing import Optional, Generic

# !Temporary constants
ROOT_DATA_PATH = '/home/brahste/Datasets/LunarAnalogue/images-screened'


class LunarAnalogueDataset(torch.utils.data.Dataset):
    """Map style dataset of the lunar analogue terrain"""
    # TODO: Consider doing data augmentation to increase the number of training samples
    def __init__(
            self,
            data_config: dict,
            train: bool = True,
            transforms: Optional[transforms.Compose] = None
    ):
        super(LunarAnalogueDataset, self).__init__()

        # We handle the training and testing data with various glob
        # patterns, this helps us be able to adapt and implement
        # alternative labelling scheme
        if train:
            self._glob_pattern = data_config['glob_pattern_train']
        else:
            self._glob_pattern = data_config['glob_pattern_test']

        self._root_data_path = Path(data_config['root_data_path'])
        self._list_of_image_paths = list(self._root_data_path.glob(self._glob_pattern))
        self._transforms = transforms

    def __len__(self):
        '''Returns the total number of images in the dataset'''
        return len(self._list_of_image_paths)

    def __getitem__(self, idx: int):

        image = io.imread(self._list_of_image_paths[idx])

        if self._transforms:
            image = self._transforms(image)

        return image

    def get_label(self, idx: int):

        if 'typical/' in str(self._list_of_image_paths[idx]):
            return 0
        elif 'novel/' in str(self._list_of_image_paths[idx]):
            return 1
        else:
            raise ValueError('Cannot find typical or novel in file path')


class LunarAnalogueDataGenerator:
    """
    For use with sklearn models and other CPU-based algorithms
    """
    def __init__(self, config: dict, stage):
        print(f'Loading {self.__class__.__name__}\n------')

        # Unpack configuration
        self._config = config
        self._batch_size = self._config['batch_size']
        # Declare preprocessing pipeline
        self._transforms = tools.PreprocessingPipeline()

        if stage == 'train' or stage == 'val':
            self._train_fraction = self._config['train_fraction']
            self._val_fraction = 1 - self._config['train_fraction']

            # Instantiate training dataset
            self._trainval_set = LunarAnalogueDataset(
                self._config,
                train=True,
                transforms=self._transforms
            )

            # Set training and validation split
            train_size = np.floor(len(self._trainval_set)*self._train_fraction).astype(int)
            val_size = np.floor(len(self._trainval_set)*self._val_fraction).astype(int)

            self._train_idxs, self._val_idxs = train_test_split(
                np.arange(len(self._trainval_set), dtype=int),
                test_size=val_size,
                train_size=train_size,
                random_state=42,
                shuffle=False
            )
            print(f'Training samples: {train_size}\nValidation samples: {val_size}\n')

        elif stage == 'test':
            # Setup testing data as well
            self._test_set = LunarAnalogueDataset(
                self._config,
                train=False,
                transforms=self._transforms
            )
            print(f'Testing samples: {len(self._test_set)}\n')

        else:
            raise ValueError('Only accepts the following stages: \'train\', \'val\', \'test\'.')

    def trainval_generator(self, stage: str):
        assert hasattr(self, '_trainval_set'), 'Need to instantiate datagenerator with stage=\'train\'.'

        if stage == 'train':
            idxs = self._train_idxs
        elif stage == 'val':
            idxs = self._val_idxs
        else:
            raise ValueError('Generator only accepts \'train\' and \'val\' stages.')

        batch_out = np.empty( (self._batch_size, *self._trainval_set[0].shape) )
        batch_nb = 0

        while batch_nb < (len(idxs) // self._batch_size):
            # Use sliding batch number approach to select which data to generate
            for i, idx in enumerate(idxs[batch_nb * self._batch_size : (batch_nb + 1) * self._batch_size]):
                batch_out[i] = self._trainval_set[idx]

            batch_nb += 1
            yield batch_out

    def test_generator(self):
        assert hasattr(self, '_test_set'), 'Need to instantiate datagenerator with stage=\'test\'.'

        batch_out = np.empty( (self._batch_size, *self._test_set[0].shape) )
        label_out = np.empty( (self._batch_size, ) )
        batch_nb = 0

        while batch_nb < (len(self._test_set) // self._batch_size):
            for i in range(self._batch_size):
                # Use sliding batch number approach to select which data to generate
                batch_out[i] = self._test_set[batch_nb * self._batch_size + i]
                label_out[i] = self._test_set.get_label(batch_nb * self._batch_size + i)

            batch_nb += 1
            yield batch_out, label_out


class LunarAnalogueDataModule(pl.core.datamodule.LightningDataModule):
    """
    For use with Pytorch models only
    """
    def __init__(self, config: dict):
        super(LunarAnalogueDataModule, self).__init__()

        # Unpack configuration
        self._config = config
        self._batch_size = self._config['batch_size']
        self._train_fraction = self._config['train_fraction']
        self._val_fraction = 1 - self._config['train_fraction']

        self._transforms = transforms.Compose([
            tools.PreprocessingPipeline(),
            transforms.ToTensor(),
        ])

    def setup(self, stage: Optional[str] = None):
        '''
        Prepare the data by cascading processing operations to be conducted
        during import
        '''
        if stage == 'fit' or stage is None:
            # Setup training and validation data for use in dataloaders
            dataset_trainval = LunarAnalogueDataset(
                self._config,
                train=True,
                transforms=self._transforms
            )
            # Calculate and save values for use in the training program
            self.num_train_samples = int(np.floor(len(dataset_trainval) * self._train_fraction))
            self.num_val_samples = len(dataset_trainval) - self.num_train_samples

            print(len(dataset_trainval), self.num_train_samples, self.num_val_samples)

            self._dataset_train, self._dataset_val = torch.utils.data.random_split(
                dataset_trainval,
                [self.num_train_samples, self.num_val_samples]
            )

        if stage == 'test' or stage is None:
            # Setup testing data as well
            self._dataset_test = LunarAnalogueDataset(
                self._config,
                train=False,
                transforms=self._transforms
            )
        return self

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset_train,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset_val,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dataset_test,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=4
        )


if __name__ == '__main__':
    pass
