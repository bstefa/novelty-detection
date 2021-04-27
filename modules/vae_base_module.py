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
            weight_decay_coefficient: float = 0.01,
            batch_size: int = 8,
            **kwargs):
        super().__init__()

        self.model = model
        self.lr = learning_rate if learning_rate is not None else 0.001  # This will be overwritten by auto find if None
        self.wd = weight_decay_coefficient if weight_decay_coefficient is not None else 0.01
        self._train_size = train_size
        self._val_size = val_size
        self._batch_size = batch_size

    def forward(self, x: Tensor, **kwargs) -> list:
        return self.model.forward(x, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

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
        self.log_dict(result, on_epoch=True, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb, criterion):
        batch_in, batch_labels = batch
        recons, mu, log_var = self.model.forward(batch_in)

        assert(callable(criterion)), "The novelty criterion must be callable."

        scores = criterion(images=batch_in, recons=recons, mu=mu, log_var=log_var)

        results = {
            'scores': scores,
            'labels': batch_labels
        }

        return results

    @property
    def version(self):
        return self.logger.version
