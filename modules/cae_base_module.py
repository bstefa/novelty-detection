import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.dtypes import *

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class CAEBaseModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 0.001,
            weight_decay_coefficient: float = 0.01):
        super(CAEBaseModule, self).__init__()

        self.model = model

        self.lr = learning_rate if learning_rate is not None else 0.001
        self.wd = weight_decay_coefficient if weight_decay_coefficient is not None else 0.01

        # Return a callable torch.nn.XLoss object
        # TODO: Add the ability to quickly change the loss function being implemented (e.g. SSIM)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch, batch_nb):

        batch_in, _ = self.handle_batch_shape(batch)

        batch_rc = self.forward(batch_in)
        loss = self.loss_function(batch_rc, batch_in)

        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}  # The returned object must contain a 'loss' key

    def validation_step(self, batch, batch_nb):

        batch_in, _ = self.handle_batch_shape(batch)

        batch_rc = self.forward(batch_in)
        loss = self.loss_function(batch_rc, batch_in)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):

        batch_in, batch_labels = self.handle_batch_shape(batch)
        batch_lt = self.model.encoder(batch_in)
        batch_rc = self.model.decoder(batch_lt)
        loss = self.loss_function(batch_rc, batch_in)

        # Calculate individual novelty scores
        batch_scores = torch.empty(len(batch_labels), dtype=torch.float)
        for x_nb, (x_rc, x_in) in enumerate(zip(batch_rc, batch_in)):
            batch_scores[x_nb] = self.loss_function(x_rc, x_in)

        self.log('test_loss', loss)
        return {
            'test_loss': loss,
            'scores': batch_scores,
            'labels': batch_labels,
            'images': {
                'batch_in': batch_in.detach(),
                'batch_rc': batch_rc.detach(),
                'batch_lt': batch_lt.detach()
            }
        }

    @staticmethod
    def handle_batch_shape(batch):
        '''
        Conducts an inplace operation to merge regions and batch size if neceassary
        '''
        batch_in, _ = batch
        assert any(len(batch_in.shape) == s for s in (4, 5)), f'Batch must have 4 or 5 dims, got {len(batch_in.shape)}'
        if len(batch_in.shape) == 5:
            batch_in = batch_in.view(-1, *batch_in.shape[2:])
        return batch_in, _

    @property
    def version(self):
        return self.logger.version
