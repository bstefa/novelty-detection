import pytorch_lightning as pl


class BaseDataModule(pl.LightningDataModule):
    def __init__(self):
        super(BaseDataModule, self).__init__()

    @property
    def train_size(self):
        assert hasattr(self, '_train_set'), 'Need to setup data before getting dataset length'
        return len(self._train_set)

    @property
    def val_size(self):
        assert hasattr(self, '_val_set'), 'Need to setup data before getting dataset length'
        return len(self._val_set)

    @property
    def test_size(self):
        assert hasattr(self, '_test_set'), 'Need to setup data before getting dataset length'
        return len(self._test_set)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def shape(self):
        conditions = [hasattr(self, '_test_set'), hasattr(self, '_train_set')]
        assert any(conditions), 'Need to call .setup before shape is available'
        try:
            return self._train_set[0][0].shape
        except AttributeError:
            return self._test_set[0][0].shape
