import os
import torch

from typing import Optional
from utils import tools

from datasets.base import BaseDataModule


class NoveltyMNISTDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_data_path: str,
            train: bool = True,
            data_transforms=None):
        super().__init__()

        if train:
            data_path = os.path.join(root_data_path, 'trainval.pt')
        else:
            data_path = os.path.join(root_data_path, 'test.pt')

        self._images, self._class_labels = torch.load(data_path)
        self._data_transforms = data_transforms

    def __len__(self):
        return len(self._class_labels)

    def __getitem__(self, idx: int):

        image = self._images[idx][None]  # Unsqueeze 2d array to carry a single channel

        if self._data_transforms:
            image = self._data_transforms(image)

        return image, self.get_label(idx)

    def get_label(self, idx: int):

        if any([self._class_labels[idx] == nb for nb in [0, 1, 3, 4, 5, 6, 8, 9]]):
            return 0
        elif any([self._class_labels[idx] == nb for nb in [2, 7]]):
            return 1
        else:
            raise ValueError('No matching class label was found')


class NoveltyMNISTDataModule(BaseDataModule):
    def __init__(
            self,
            root_data_path: str,
            batch_size: int = 8,
            train_fraction: float = 0.8,
            **kwargs):
        super().__init__()

        self._root_data_path = root_data_path
        self._batch_size = batch_size if batch_size is not None else 8
        self._train_fraction = train_fraction if train_fraction is not None else 0.8
        self._data_transforms = tools.NoveltyMNISTPreprocessingPipeline()

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit' or stage == 'train' or stage is None:
            trainval_set = NoveltyMNISTDataset(
                self._root_data_path,
                train=True,
                data_transforms=self._data_transforms)

            train_size = int(torch.floor(len(trainval_set) * self._train_fraction))
            val_size = len(trainval_set) - train_size
            # Since setup is called from every process, setting state here is okay
            self._train_set, self._val_set = torch.utils.data.random_split(trainval_set, [train_size, val_size])

        elif stage == 'test' or stage is None:
            self._test_set = NoveltyMNISTDataset(
                self._root_data_path,
                train=False,
                data_transforms=self._data_transforms)

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
