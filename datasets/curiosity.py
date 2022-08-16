"""
Classes used to import and train models on multi-spectral images from Curiosity.
Download the data from: https://zenodo.org/record/3732485#.YFTm9P5E2V4

Assumes data is structured as follows:
    <root_data_path>/
        trainval/
            train_typical/
            validation_typical/
        test/
            test_typical/
            test_novel/
                all/
                ...<novelty subclasses (eg. bedrock)>/

Author: Braden Stefanuk
Created: Mar. 18, 2021
"""
import torch
import numpy as np

from utils import tools
from datasets.base import BaseDataModule
from utils.dtypes import *


class CuriosityDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_data_path: str,
            train: bool = True,
            data_transforms=None,
            novel_class_specifier='test_novel/all/'):
        super().__init__()

        if train:
            self._list_of_image_paths = tools.PathGlobber(root_data_path).glob('trainval/**/*.npy')
        else:
            self._list_of_image_paths = tools.PathGlobber(root_data_path).multiglob(
                (f'test/{p}*.npy' for p in ['test_typical/', novel_class_specifier]))
        self._train = train
        self._data_transforms = data_transforms

    def __len__(self):
        return len(self._list_of_image_paths)

    def __getitem__(self, idx: int):
        image = np.load(self._list_of_image_paths[idx]).astype(np.float32)
        if self._data_transforms:
            image = self._data_transforms(image)

        return image, self.get_label(idx)

    def get_label(self, idx: int):
        if 'typical' in str(self._list_of_image_paths[idx]):
            return 0
        elif 'novel' in str(self._list_of_image_paths[idx]):
            return 1
        else:
            raise ValueError('Cannot find typical/ or novel/ in file path')

    def get_split(self) -> Tuple[np.ndarray, np.ndarray]:
        dummy_image, _ = self[0]
        all_images = np.empty((len(self), *dummy_image.shape))
        all_labels = np.empty((len(self), ))
        for i in range(len(self)):
            image, label = self[i]
            all_images[i] = image.numpy()
            all_labels[i] = label
        return all_images, all_labels


class CuriosityDataModule(BaseDataModule):
    def __init__(
            self,
            root_data_path: str,
            data_transforms: Compose,
            batch_size: int = 8,
            train_fraction: float = 0.85,
            **kwargs):
        super().__init__()

        # Handle NoneTypes (defaults) passed from configuration file
        self._batch_size = batch_size if batch_size is not None else 8
        self._train_fraction = train_fraction if train_fraction is not None else 0.85
        self._root_data_path = root_data_path
        self._val_fraction = 1 - self._train_fraction

        assert (data_transforms is not None), \
            'Data transforms must be defined in the training script. See utils/__init__.py.'
        self._data_transforms = data_transforms

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):

        if stage == 'fit' or stage == 'train' or stage is None:
            # Setup training and validation data for use in dataloaders
            trainval_set = CuriosityDataset(
                self._root_data_path,
                train=True,
                data_transforms=self._data_transforms)
            # Since setup is called from every process, setting state here is okay
            train_size = int(np.floor(len(trainval_set) * self._train_fraction))
            val_size = len(trainval_set) - train_size

            self._train_set, self._val_set = torch.utils.data.random_split(trainval_set, [train_size, val_size])

        if stage == 'test' or stage is None:
            # Setup testing data as well
            self._test_set = CuriosityDataset(
                self._root_data_path,
                train=False,
                data_transforms=self._data_transforms)

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
            drop_last=False,
            num_workers=8)

    def split(self, train: bool) -> Tuple[np.ndarray, np.ndarray]:
        ds = CuriosityDataset(self._root_data_path, train=train, data_transforms=self._data_transforms)
        return ds.get_split()
