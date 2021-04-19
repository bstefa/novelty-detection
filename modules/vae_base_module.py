import torch.nn as nn
import pytorch_lightning as pl

from utils.dtypes import *


class VAEBaseModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            train_size: int = 0,
            val_size: int = 0,
            learning_rate: float = 0.001,
            batch_size: int = 8,
            **kwargs):
        super().__init__()

        self.model = model
        self.lr = learning_rate if learning_rate is not None else 0.001  # This will be overwritten by auto find if None
        self._train_size = train_size
        self._val_size = val_size
        self._batch_size = batch_size

    def forward(self, x: Tensor, **kwargs) -> list:
        return self.model.forward(x, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        images, labels = batch
        # Results is a list of outputs computed during the forward pass
        # of the form: [reconstructions, input, mu, log_var]
        results_list = self.forward(images, labels=labels)

        # Note that train_loss is a dictionary containing the various losses
        train_loss_dict = self.model.loss_function(
            *results_list,
            M_N=self._batch_size / self._train_size,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx)

        self.log_dict(train_loss_dict)
        return train_loss_dict

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        results_list = self.forward(real_img, labels=labels)
        val_loss_dict = self.model.loss_function(
            *results_list,
            M_N=self._batch_size / self._val_size,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx)

        return val_loss_dict

    def validation_epoch_end(self, outputs):
        avg_elbo_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recons_loss = torch.stack([x['reconstruction_loss'] for x in outputs]).mean()
        avg_kld_loss = torch.stack([x['KLD'] for x in outputs]).mean()

        self.log('avg_elbo_loss', avg_elbo_loss)
        self.log('avg_recons_loss', avg_recons_loss)
        self.log('avg_kld_loss', avg_kld_loss)

        pass

    def test_step(self, batch, batch_nb):
        batch_in, batch_labels = batch
        recons = self.model.generate(batch_in)

        mse_loss = torch.nn.MSELoss(reduction='none')
        recons_error = mse_loss(recons, batch_in)
        mse_loss_sum = torch.sum(recons_error, dim=(1, 2, 3))

        results_dict = {
            'scores': mse_loss_sum,
            'labels': batch_labels
        }

        return results_dict

    @property
    def version(self):
        return self.logger.version
