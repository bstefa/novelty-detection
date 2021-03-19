import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.dtypes import *

class CAEBaseModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 0.001,
            weight_decay_coefficient: float = 0.01):
        super(CAEBaseModule, self).__init__()

        self.model = model
        self.lr = learning_rate if learning_rate is not None else 0.001
        self.wd = weight_decay_coefficient

        # Return a callable torch.nn.XLoss object
        # TODO: Add the ability to quickly change the loss function being implemented (e.g. SSIM)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch, batch_nb):

        batch_in, _ = batch

        batch_rc = self.forward(batch_in)
        loss = self.loss_function(batch_rc, batch_in)

        self.log('train_loss', loss, on_epoch=True)
        return {'loss': loss}  # The returned object must contain a 'loss' key

    def validation_step(self, batch, batch_nb):

        batch_in, _ = batch

        batch_rc = self.forward(batch_in)
        loss = self.loss_function(batch_rc, batch_in)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):

        batch_in, batch_labels = batch
        batch_rc = self.forward(batch_in)
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
                'batch_in': batch_in.detach(),  # Tensor
                'batch_rc': batch_rc.detach()  # Tensor
            }
        }

    @property
    def version(self):
        return self.logger.version
