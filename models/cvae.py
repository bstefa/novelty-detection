import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os

class VariationalAutoEncoder(pl.LightningModule):

    def __init__(self,
                in_channels: int,
                latent_dims: int,
                hidden_dims: List = None,
                input_height: int,
                input_width: int):

        super().__init__()

        self.latent_dims = latent_dims

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 
                              out_channels=h_dim,
                              kernel_size= 3, 
                              stride= 2, 
                              padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # TODO: check output dims of encoder
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dims)

        # Build Decoder
        modules = []

        # TODO: check input dims for decoder
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


    def encode(self, input: Tensor) -> List[Tensor]:
        """
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
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparametrization trick

        z = mu + std*error
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]

        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)

        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Compute ELBO loss
        """

        reconstruction = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(reconstruction, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        elbo_loss = reconstruction_loss + kld_weight * kld_loss

        return {'loss': loss, 'reconstruction_loss':reconstruction_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Sample from latent space
        """

        z = torch.randn(num_samples, self.latent_dims)
        z = z.to(current_device)

        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Sample from q(z|x) and return reconstruction
        """

        return self.forward(x)[0]
