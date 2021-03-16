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

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.discriminator = model.discriminator
        self.prior = model.prior

        self.lr = learning_rate if learning_rate != 'auto' else 0.001
        self._train_size = train_size
        self._val_size = val_size
        self._batch_size = batch_size
        self._data_shape

    def configure_optimizers(self):
        opt_reconstruction = torch.optim.Adam(self)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # See github.com/brahste/novelty-detection/figures/SimpleAAESchematic for an overview of this AAE

        # Manage data layout
        batch_in, _ = batch
        batch_in = batch_in.view(self._batch_size, -1)  # Keep batch dimension, but flatten all others

        # Reconstruction phase
        # ------
        z_generated = self.encoder(batch_in)
        batch_rc = self.model.decoder(z_generated)

        # TODO: Set up seperate loss functions for autoencoder and discriminator
        reconstruction_loss = nn.MSELoss()(batch_rc, batch_in)

        # Regularization phase
        # ------
        self.model.encoder.eval()  # Freeze dropout and other non-training parameters

        # Get the probabilities that each sample was taken from the prior
        dscrm_prob_latent = self.discriminator(self.model.encoder(batch_in))
        dscrm_prob_prior = self.discriminator(self.model.prior(*batch_in.shape))

        discriminator_loss = -torch.mean(torch.log(dscrm_prob_prior) + torch.log(1 - dscrm_prob_latent))

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        results_list = self.forward(real_img, labels=labels)
        val_loss_dict = self.model.loss_function(
            *results_list,
            M_N=self._batch_size / self._val_size,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx)

        return val_loss_dict

    def generator_loss(self, batch_in: Tensor):

        z_generated = self.encoder(batch_in)
        return -torch.mean(torch.log(self.discriminator(z_generated)))

    def discriminator_loss(self, batch_in: Tensor):

        z_real = self.model.prior(*batch_in.shape, device=self.device)
        d_loss_real = self.model.discriminator(z_real)

        z_fake = self.model.encoder(batch_in)
        d_loss_fake = self.model.discriminator(z_fake)
        # TODO: Refamiliarize yourself with the concept of log-likelihood
        return -torch.mean(torch.log(d_loss_real) + torch.log(1 - d_loss_fake))

    def reconstruction_loss(self, batch_in: Tensor):

        # TODO: Implement additional loss functions here
        loss_fn = nn.MSELoss()
        return loss_fn(self.decoder(self.encoder(batch_in)))

    @property
    def version(self):
        return self.logger.version
