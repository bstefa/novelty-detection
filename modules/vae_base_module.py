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

    def training_step(self, batch, batch_idx):

        batch_in, _ = batch
        batch_rc, mu, log_var = self.forward(batch_in)

        elbo_loss, rc_loss, kld_loss = self.model.loss_function(
            batch_rc, batch_in, mu, log_var, M_N=self._batch_size / self._train_size)

        result = {'loss': elbo_loss, 'rc_loss': rc_loss, 'kld_loss': kld_loss}
        self.log_dict(result)
        return result  # Needs 'loss' key in return dict

    def validation_step(self, batch, batch_idx):
        batch_in, _ = batch

        batch_rc, mu, log_var = self.forward(batch_in)

        elbo_loss, rc_loss, kld_loss = self.model.loss_function(
            batch_rc, batch_in, mu, log_var, M_N=self._batch_size / self._train_size)

        result = {'val_elbo_loss': elbo_loss, 'val_rc_loss': rc_loss, 'val_kld_loss': kld_loss}
        self.log_dict(result, on_epoch=True)
        return result

    def validation_epoch_end(self, outputs):
        # TODO: This may be unneeded after specifying 'on_epoch=True' in validation step
        avg_elbo_loss = torch.stack([x['val_elbo_loss'] for x in outputs]).mean()
        avg_recons_loss = torch.stack([x['val_rc_loss'] for x in outputs]).mean()
        avg_kld_loss = torch.stack([x['val_kld_loss'] for x in outputs]).mean()
        self.log('avg_elbo_loss', avg_elbo_loss)
        self.log('avg_recons_loss', avg_recons_loss)
        self.log('avg_kld_loss', avg_kld_loss)

    def test_step(self, batch, batch_nb):
        batch_in, batch_labels = batch
        recons = self.model.generate(batch_in)

        mse_loss = torch.nn.MSELoss(reduction='none')
        recons_error = mse_loss(recons, batch_in)
        mse_loss_sum = torch.sum(recons_error, dim=(1, 2, 3))

        results = {
            'scores': mse_loss_sum,
            'labels': batch_labels
        }
        return results

    @property
    def version(self):
        return self.logger.version
