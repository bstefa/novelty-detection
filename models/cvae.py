import torch

from torch import nn
from torch.nn import functional as F
from utils import tools
from utils.dtypes import *


class VariationalAutoEncoder(nn.Module):

    def __init__(self, in_shape: torch.Tensor, latent_dims: int, **kwargs):
        super(VariationalAutoEncoder, self).__init__()

        self.C, self.H, self.W = in_shape
        self._latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Conv2d(self.C, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        # TODO: check output dims of encoder
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dims)
        self.fc_var = nn.Linear(64 * 4 * 4, latent_dims)

        # TODO: check input dims for decoder
        self.decoder_input = nn.Linear(latent_dims, 64 * 4 * 4)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.final_layer = nn.Sequential(

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Receives an input image and outputs the mean and the logvar for the distribution q(z|x)
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Receives a variable sampled from q(z) and transforms it back to p(x|z)
        """
        result = self.decoder_input(z)
        result = result.view(-1, 64, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparametrization trick to enable back propagation while maintaining 
        stochasticity by injecting a randomly sampled variable 

        z = mu + std*eps
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Receives an input image and returns the reconstruction, input image,
        and the mean and logvar of the Gaussian q(z|x)
        """

        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)

        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Compute Evidence Lower Bound (elbo) loss

        elbo = reconstruction loss + Kullback-Leibler divergence
        """

        reconstruction = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        reconstruction_loss = F.mse_loss(reconstruction, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        elbo_loss = reconstruction_loss + kld_weight * kld_loss

        return {'loss': elbo_loss, 'reconstruction_loss': reconstruction_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Sample z from latent space q(z) and return reconstruction p(x|z)
        """

        z = torch.randn(num_samples, self._latent_dims)
        z = z.to(current_device)

        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Sample from q(z|x) and return reconstruction p(x|z)
        """

        return self.forward(x)[0]
