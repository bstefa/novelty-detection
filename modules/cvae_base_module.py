import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import pytorch_lightning as pl
import torchvision

from utils.dtypes import *


class CVAEBaseModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            train_size: int,
            val_size: int,
            learning_rate: float = 0.001,
            batch_size: int = 8,
            **kwargs):
        super(CVAEBaseModule, self).__init__()

        self.model = model
        self.lr = learning_rate if learning_rate != 'auto' else 0.001
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

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        # self.sample_images()
        return {'val_loss': avg_loss}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.dm.val_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)

        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass

        del test_input, recons  # , samples

    @property
    def version(self):
        return self.logger.version
