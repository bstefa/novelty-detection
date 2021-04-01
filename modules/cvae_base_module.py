import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import pytorch_lightning as pl
import torchvision

from utils.dtypes import *
import torchvision.transforms.functional as TF

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

        if batch_idx == 0 and self.version is not None:
            index = torch.randint(len(real_img), (1,1)).numpy()
            self.sample_images(real_img[index], labels[index])

        return val_loss_dict

    def validation_epoch_end(self, outputs):
        avg_elbo_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recons_loss = torch.stack([x['reconstruction_loss'] for x in outputs]).mean()
        avg_kld_loss = torch.stack([x['KLD'] for x in outputs]).mean()

        self.log('avg_elbo_loss', avg_elbo_loss)
        self.log('avg_recons_loss', avg_recons_loss)
        self.log('avg_kld_loss', avg_kld_loss)

        pass

    def sample_images(self, real_img: Tensor, label: Tensor):
        # Get sample reconstruction image

        recons = self.model.generate(real_img)
        
        real_img = TF.rotate(real_img, -90)
        recons = TF.rotate(recons, -90)
        real_img = TF.hflip(real_img)
        recons = TF.hflip(recons)

        grid = torch.cat((real_img, recons))

        vutils.save_image(grid,
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=1)

        samples = self.model.sample(144, next(self.model.parameters()).device)

        samples = TF.rotate(samples, -90)
        samples = TF.hflip(samples)

        vutils.save_image(samples,
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        del recons

    @property
    def version(self):
        return self.logger.version
