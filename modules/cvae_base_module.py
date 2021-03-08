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
            data_module,
            model,
            hparams: dict):
        super(CVAEBaseModule, self).__init__()

        self.dm = data_module
        self.model = model
        self.hparams = hparams
        self.curr_device = None

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        images, labels = batch
        self.curr_device = images.device

        results = self.forward(images, labels=labels)
        train_loss = self.model.loss_function(*results,
                                            M_N = self.hparams['batch_size']/self.dm.len_train_data,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.hparams['batch_size']/self.dm.len_val_data,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.dm.val_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)

        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass


        del test_input, recons #, samples

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def test_dataloader(self):
        return self.dm.test_dataloader()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                               lr=self.hparams['learning_rate'])



