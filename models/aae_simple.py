import torch
import torch.nn as nn

from utils.dtypes import *


class Encoder(nn.Module):
    def __init__(self, in_nodes: int, latent_nodes: int = 8):
        super(Encoder, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.encoder = nn.Sequential(
            nn.Linear(in_nodes, 1000),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(1000, latent_nodes))

    def forward(self, x):
        # print('ENC INPUT ', x.shape)
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, out_nodes: int, latent_nodes: int = 8):
        super(Decoder, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.decoder = nn.Sequential(
            nn.Linear(latent_nodes, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, out_nodes),
            nn.Sigmoid())

    def forward(self, x):
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, latent_nodes: int = 8):
        super(Discriminator, self).__init__()

        # TODO: Consider initializing the weights explicitly here
        self.discriminator = nn.Sequential(
            nn.Linear(latent_nodes, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.discriminator(x)


class SimpleAAE(nn.Module):
    def __init__(self, in_nodes: int, latent_nodes: int = 8):
        super().__init__()

        self.encoder = Encoder(in_nodes=in_nodes, latent_nodes=latent_nodes)
        self.decoder = Decoder(out_nodes=in_nodes, latent_nodes=latent_nodes)
        self.discriminator = Discriminator(latent_nodes=latent_nodes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prior(self, *size):
        return torch.randn(*size, device=self.device)

    def generator_loss(self, batch_in: Tensor, epsilon: float = 1e-8):
        self.encoder.train()
        self.discriminator.eval()

        z_generated = self.encoder(batch_in)
        d_score = self.discriminator(z_generated)

        assert not any(torch.isnan(d_score)), 'Can\'t pass NaN to logarithm'
        assert not any(d_score < 0.), 'Can\'t pass negative number to logarithm'

        return -torch.mean(torch.log(d_score + epsilon))

    def discriminator_loss(self, batch_in: Tensor, epsilon: float = 1e-8):
        self.discriminator.train()
        self.encoder.eval()  # Freeze dropout and other non-training parameters

        z_fake = self.encoder(batch_in)
        d_score_fake = self.discriminator(z_fake)

        # The discriminator take samples with size of the latent space
        z_real = self.prior(*z_fake.shape)
        d_score_real = self.discriminator(z_real)

        assert not any(d_score_fake < 0.), 'Abort. Can\'t pass negative number to logarithm'
        assert not any(d_score_real < 0.), 'Abort. Can\'t pass negative number to logarithm'
        # TODO: Replace this loss with losses already implement in Pytorch is possible
        return -torch.mean(torch.log(d_score_real + epsilon) + torch.log(1 - d_score_fake + epsilon))

    def reconstruction_loss(self, batch_in: Tensor):
        self.encoder.train()
        self.decoder.train()

        # TODO: Implement additional loss functions here
        loss_fn = nn.MSELoss()
        batch_rc = self.decoder(self.encoder(batch_in))
        return loss_fn(batch_rc, batch_in)
