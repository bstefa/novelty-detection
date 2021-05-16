import os
import torch
import numpy as np

from utils.dtypes import *

from datasets.base import BaseDataModule


class NoveltyMNISTDataset(torch.utils.data.Dataset):
    """
    Fundemental class for importing Novelty MNIST data. This class should *never* be instantiated directly. Any
    calls to the class should be handled implicitly through the associated DataModule.
    """
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

    def get_split(self) -> Tuple[np.ndarray, np.ndarray]:
        novelty_labels = np.empty(len(self._class_labels))
        for i in range(len(novelty_labels)):
            novelty_labels[i] = self.get_label(i)
        images = self._images.numpy()

        return images, novelty_labels


class NoveltyMNISTDataModule(BaseDataModule):
    def __init__(
            self,
            root_data_path: str,
            data_transforms: Compose,
            batch_size: int = 8,
            train_fraction: float = 0.8,
            **kwargs):
        super().__init__()

        self._root_data_path = root_data_path
        self._batch_size = batch_size if batch_size is not None else 8
        self._train_fraction = train_fraction if train_fraction is not None else 0.8

        assert (data_transforms is not None), \
            'Changes have been made. Data transforms must now be defined in the training script. See utils/__init__.py.'
        self._data_transforms = data_transforms

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Since setup is called from every process, setting state here is okay
        if stage == 'fit' or stage == 'train' or stage is None:
            self._trainval_set = NoveltyMNISTDataset(
                self._root_data_path,
                train=True,
                data_transforms=self._data_transforms)

            train_size = int(len(self._trainval_set) * self._train_fraction)
            val_size = len(self._trainval_set) - train_size
            self._train_set, self._val_set = torch.utils.data.random_split(
                self._trainval_set, [train_size, val_size])

        elif stage == 'test':
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
        ds = NoveltyMNISTDataset(self._root_data_path, train=train)
        return ds.get_split()
