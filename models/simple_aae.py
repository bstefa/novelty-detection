import torch
import torch.nn as nn

from utils.dtypes import *


class Encoder(nn.Module):
    def __init__(self, in_chans: int, latent_chans: int = 8):
        super(Encoder, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.encoder = nn.Sequential(
            nn.Linear(in_chans, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, latent_chans))

    def forward(self, x):
        # print('ENC INPUT ', x.shape)
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, out_chans: int, latent_chans: int = 8):
        super(Decoder, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.decoder = nn.Sequential(
            nn.Linear(latent_chans, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, out_chans),
            nn.Sigmoid())

    def forward(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, latent_chans: int = 8):
        super(Discriminator, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.discriminator = nn.Sequential(
            nn.Linear(latent_chans, 1000),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.discriminator(x)


class SimpleAAE(nn.Module):
    def __init__(self, in_chans: int, latent_chans: int = 8, **kwargs):
        super().__init__()

        self.encoder = Encoder(in_chans=in_chans, latent_chans=latent_chans)
        self.decoder = Decoder(out_chans=in_chans, latent_chans=latent_chans)
        self.discriminator = Discriminator(latent_chans=latent_chans)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prior(self, *size):
        return torch.randn(*size, device=self.device)

    def generator_loss(self, batch_in: Tensor):
        self.encoder.train()
        self.discriminator.eval()

        z_generated = self.encoder(batch_in)
        return -torch.mean(torch.log(self.discriminator(z_generated)))

    def discriminator_loss(self, batch_in: Tensor):
        self.discriminator.train()
        self.encoder.eval()  # Freeze dropout and other non-training parameters

        z_fake = self.encoder(batch_in)
        d_loss_fake = self.discriminator(z_fake)

        # The discriminator take samples with size of the latent space
        z_real = self.prior(*z_fake.shape)
        d_loss_real = self.discriminator(z_real)

        # TODO: Refamiliarize yourself with the concept of log-likelihood
        return -torch.mean(torch.log(d_loss_real) + torch.log(1 - d_loss_fake))

    def reconstruction_loss(self, batch_in: Tensor):
        self.encoder.train()
        self.decoder.train()

        # TODO: Implement additional loss functions here
        loss_fn = nn.MSELoss()
        batch_rc = self.decoder(self.encoder(batch_in))
        return loss_fn(batch_rc, batch_in)
