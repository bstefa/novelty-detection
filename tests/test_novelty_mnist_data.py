import unittest
import torch
import pytorch_lightning as pl

from pathlib import Path
from utils import tools
from datasets import supported_datamodules
from models.cae_baseline import BaselineCAE


class TestNoveltyMNISTData(unittest.TestCase):

    def setUp(self):
        self.cae_config = tools.config_from_file('configs/cae/cae_baseline_mnist.yaml')
        self.dm = supported_datamodules['NoveltyMNISTDataModule'](**self.cae_config['data-parameters'])

    def test_datamodule_instantiation(self):
        self.assertTrue(Path(self.dm._root_data_path).is_dir())

        self.assertIsInstance(self.dm._batch_size, int)
        self.assertGreater(self.dm._batch_size, 1)

        self.assertIsInstance(self.dm._train_fraction, float)
        self.assertGreater(self.dm._train_fraction, 0.5)
        self.assertLess(self.dm._train_fraction, 1.0)

    def test_cae_datamodule_setup(self):
        self.dm.prepare_data()
        self.dm.setup('train')
        self.dm.setup('test')

        self.assertEqual(self.dm.data_shape, (1, 28, 28))  # This data shape is known from the MNIST set
        self.assertGreater(self.dm.train_size/self.dm.batch_size, self.dm.val_size/self.dm.batch_size)

        for tr, vl, ts in zip(self.dm.train_dataloader(), self.dm.val_dataloader(), self.dm.test_dataloader()):
            tr_data, tr_label = tr
            vl_data, vl_label = vl
            ts_data, ts_label = ts
            self.assertTrue(all((len(dshape) == 4 for dshape in (tr_data.shape, vl_data.shape, ts_data.shape))))
            self.assertTrue(all((dtype is torch.float32 for dtype in (tr_data.dtype, vl_data.dtype, ts_data.dtype))))
            self.assertTrue(all((dtype is torch.int64 for dtype in (tr_label.dtype, vl_label.dtype, ts_label.dtype))))

    def test_data_with_cae_baseline_model(self):
        self.dm.prepare_data()
        self.dm.setup('train')
        self.dm.setup('test')

        model = BaselineCAE(in_chans=self.dm.data_shape[0])

        def test_x_step(dataloader):
            data, label = next(iter(dataloader))

            z = model.encoder(data)
            self.assertEqual(z.shape, torch.Size([256, 3, 7, 7]))  # This will change if the architecture changes

            x_hat = model.decoder(z)
            self.assertEqual(x_hat.shape, data.shape)
            self.assertEqual(x_hat.dtype, data.dtype)

        test_x_step(self.dm.train_dataloader())
        test_x_step(self.dm.val_dataloader())
        test_x_step(self.dm.test_dataloader())

    # TODO: Write test for other models (AAE, VAE, PCA)


if __name__ == '__main__':
    unittest.main()
