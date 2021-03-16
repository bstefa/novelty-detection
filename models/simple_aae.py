import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_chans: int, latent_chans: int = 8):
        super(Encoder, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.encoder = nn.Sequential(
            nn.Linear(in_chans, 1000),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.1),
            nn.RelU(),
            nn.Linear(1000, latent_chans))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, out_chans: int, latent_chans: int = 8):
        super(Decoder, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.encoder = nn.Sequential(
            nn.Linear(latent_chans, 1000),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.1),
            nn.RelU(),
            nn.Linear(1000, out_chans),
            nn.Tanh())

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
            nn.RelU(),
            nn.Linear(1000, 1),
            nn.Tanh())

    def forward(self, x):
        return self.discriminator(x)


class SimpleAAE(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, latent_chans: int = 8, **kwargs):
        super(SimpleAAE, self).__init__()

        self.encoder = Encoder(in_chans, latent_chans)
        self.decoder = Decoder(out_chans, latent_chans)
        self.discriminator = Discriminator(latent_chans)

    @staticmethod
    def prior(*size, device):
        return torch.randn(*size, device=device)
