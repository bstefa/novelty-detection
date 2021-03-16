import torch
import torch.nn as nn
import torchvision.utils as vutils
import pytorch_lightning as pl

from utils.dtypes import *


class AAEBaseModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            train_size: int,
            val_size: int,
            learning_rate: float = 0.001,
            batch_size: int = 8,
            **kwargs):
        super().__init__()

        # The model must implement the following methods and attributes
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.discriminator = model.discriminator
        self.discriminator_loss = model.discriminator_loss
        self.reconstruction_loss = model.reconstruction_loss
        self.generator_loss = model.generator_loss

        self.lr = learning_rate if learning_rate != 'auto' else 0.001
        self._train_size = train_size
        self._val_size = val_size
        self._batch_size = batch_size

    def configure_optimizers(self):
        opt_reconstruction = torch.optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()])
        opt_generator = torch.optim.Adam(self.encoder.parameters())
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters())
        return [opt_reconstruction, opt_discriminator, opt_generator], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        # See github.com/brahste/novelty-detection/figures/SimpleAAESchematic

        # Manage data layout
        batch_in, _ = batch
        batch_in = batch_in.view(self._batch_size, -1)  # Keep batch dimension, but flatten all others

        # Reconstruction phase
        # ------
        if optimizer_idx == 0:
            reconstruction_loss = self.reconstruction_loss(batch_in)
            self.log('r_loss', reconstruction_loss, on_epoch=True, prog_bar=True)
            return reconstruction_loss

        # Regularization phase
        # ------
        if optimizer_idx == 1:
            discriminator_loss = self.discriminator_loss(batch_in)
            self.log('d_loss', discriminator_loss, on_epoch=True, prog_bar=True)
            return discriminator_loss

        if optimizer_idx == 2:
            generator_loss = self.generator_loss(batch_in)
            self.log('g_loss', generator_loss, on_epoch=True, prog_bar=True)
            return generator_loss

    @property
    def version(self):
        return self.logger.version
