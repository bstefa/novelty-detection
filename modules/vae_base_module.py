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

        batch_in, _ = self.handle_batch_shape(batch)
        batch_rc, mu, log_var = self.forward(batch_in)

        elbo_loss, rc_loss, kld_loss = self.model.loss_function(
            batch_rc, batch_in, mu, log_var, M_N=self._batch_size / self._train_size)

        result = {'loss': elbo_loss, 'rc_loss': rc_loss, 'kld_loss': kld_loss}
        self.log_dict(result)
        return result  # Needs 'loss' key in return dict

    def validation_step(self, batch, batch_idx):
        batch_in, _ = self.handle_batch_shape(batch)

        batch_rc, mu, log_var = self.forward(batch_in)

        elbo_loss, rc_loss, kld_loss = self.model.loss_function(
            batch_rc, batch_in, mu, log_var, M_N=self._batch_size / self._train_size)

        result = {'val_elbo_loss': elbo_loss, 'val_rc_loss': rc_loss, 'val_kld_loss': kld_loss}
        self.log_dict(result, on_epoch=True, prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        batch_in, batch_labels = self.handle_batch_shape(batch)
        image_shape = batch_in.shape
        batch_rc, mu, log_var, batch_lt = self.model.generate_for_testing(batch_in)

        # Calculate individual novelty scores
        batch_scores = []
        for x_nb, (x_rc, x_in) in enumerate(zip(batch_rc, batch_in)):
            batch_scores.append(nn.MSELoss()(x_rc, x_in))

        results = {
            'scores': batch_scores,
            'labels': batch_labels,
            'recons': batch_rc,
            'images': {
                'batch_in': batch_in.detach().reshape(*image_shape),
                'batch_rc': batch_rc.detach().reshape(*image_shape),
                'batch_lt': batch_lt.detach()
            }
        }
        return results

    @staticmethod
    def handle_batch_shape(batch):
        """
        Conducts an inplace operation to merge regions and batch size if neceassary
        """
        batch_in, _ = batch
        assert any(len(batch_in.shape) == s for s in (4, 5)), f'Batch must have 4 or 5 dims, got {len(batch_in.shape)}'
        if len(batch_in.shape) == 5:
            batch_in = batch_in.view(-1, *batch_in.shape[2:])
        return batch_in, _

    @property
    def version(self):
        return self.logger.version
