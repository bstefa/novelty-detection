from utils.dtypes import *
import pytorch_lightning as pl


class BaseDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def prepare_date(self) -> None:
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def train_size(self):
        assert hasattr(self, '_train_set'), 'Need to setup data before getting dataset length'
        return len(self._train_set)

    @property
    def val_size(self) -> int:
        assert hasattr(self, '_val_set'), 'Need to setup data before getting dataset length'
        return len(self._val_set)

    @property
    def test_size(self) -> int:
        assert hasattr(self, '_test_set'), 'Need to setup data before getting dataset length'
        return len(self._test_set)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def data_shape(self):
        conditions = [hasattr(self, '_test_set'), hasattr(self, '_train_set')]
        assert any(conditions), 'Need to call .setup before shape is available'
        try:
            # First [0] accesses image from (image,label) tuple, second [0] accesses batch element
            data_shape = self._train_set[0][0].shape
        except AttributeError:
            data_shape = self._test_set[0][0].shape

        if len(data_shape) == 3:
            return data_shape
        elif len(data_shape) == 4:
            return data_shape[len(data_shape)-3:]
        else:
            raise ValueError('Only data shapes with 3 or 4 dimensions are accepted')
