from torch import nn
from torch.nn import functional as F
from utils.dtypes import *
from utils import tools
from models.blocks import EncodingBlock, DecodingBlock


class ParentVAE(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparametrization trick to enable back propagation while maintaining
        stochasticity by injecting a randomly sampled variable

        z = mu + std*eps
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: Tensor) -> tuple:
        """
        Receives an input image (x) and outputs the mean (mu) and log-variance (log_var) for the distribution q(z|x)
        """
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)

        # Split the result into mean (mu) and variance (log_var) of the latent Gaussian distribution
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        return mu, log_var

    def decode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple:
        """
        Receives an input image and returns the reconstruction, input image,
        and the mean and logvar of the Gaussian q(z|x)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    @staticmethod
    def loss_function(batch_rc: Tensor, batch_in: Tensor, mu: Tensor, log_var: Tensor, **kwargs) -> tuple:
        """
        Compute Evidence Lower Bound (elbo) loss
        elbo = reconstruction loss + Kullback-Leibler divergence
        """
        assert (batch_in.shape == batch_rc.shape), \
            f'Input and reconstruction shapes don\'t match, got {batch_in.shape}, {batch_rc.shape}'
        reconstruction_loss = F.mse_loss(batch_rc, batch_in)

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        elbo_loss = reconstruction_loss + kld_weight * kld_loss

        return elbo_loss, reconstruction_loss, -kld_loss

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Sample z from latent space q(z) and return reconstruction p(x|z)
        """
        z = torch.randn(num_samples, self._latent_nodes, device=torch.device('cuda'))
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Sample from q(z|x) and return reconstruction p(x|z)
        """
        x_hat, _, _ = self.forward(x)
        return x_hat


class BaselineVAE(ParentVAE):

    def __init__(self, in_shape: torch.Tensor, latent_nodes: int, **kwargs):
        super().__init__()

        in_chans = in_shape[0]
        height, width = int(in_shape[1]), int(in_shape[2])
        assert isinstance(in_chans, int) and isinstance(height, int), \
            f'in_chans must be of type int, got {type(in_chans)}, and {type(height)}'
        assert any((in_chans == nb for nb in [1, 3, 6])), \
            f'Input image must be greyscale (1), RGB/YUV (3), or 6-channel multispectral, got {in_chans} channels'
        assert any((height == nb for nb in [28, 64, 248])), \
            f'Input image must have height == 28, 64, or 248, got {height}'

        self.encoder_output_height = tools.output_shape_conv2d2(
            height,
            padding=[2]*7,
            kernel_size=[5]*7,
            stride=[1, 1, 2, 1, 1, 2, 1])

        self.encoder_output_width = tools.output_shape_conv2d2(
            width,
            padding=[2]*7,
            kernel_size=[5]*7,
            stride=[1, 1, 2, 1, 1, 2, 1])

        self._latent_nodes = latent_nodes

        self.encoder = nn.Sequential(
            EncodingBlock(in_chans, 24),
            EncodingBlock(24, 48),
            EncodingBlock(48, 48, stride=2),
            EncodingBlock(48, 24),
            EncodingBlock(24, 16),
            EncodingBlock(16, 8, stride=2),
            nn.Conv2d(8, 3, kernel_size=5, padding=2),
        )

        # use 3x multiplier here because the encoder outputs 3 channels
        self.fc_mu = nn.Linear(
            3 * self.encoder_output_height*self.encoder_output_width, latent_nodes)
        self.fc_var = nn.Linear(
            3 * self.encoder_output_height*self.encoder_output_width, latent_nodes)

        self.decoder_input = nn.Linear(
            latent_nodes, 3 * self.encoder_output_height*self.encoder_output_width)

        self.decoder = nn.Sequential(
            DecodingBlock(3, 8),
            DecodingBlock(8, 16, stride=2, output_padding=(1, 0) if height == 248 else 1),
            DecodingBlock(16, 24),
            DecodingBlock(24, 48),
            DecodingBlock(48, 48, stride=2, output_padding=(1, 0) if height == 248 else 1),
            DecodingBlock(48, 24),
            nn.Conv2d(24, in_chans, kernel_size=5, padding=2),
            nn.Tanh()  # Same size as input
        )

    def decode(self, z: Tensor) -> Tensor:
        """
        Receives a variable sampled from q(z) and transforms it back to p(x|z)
        """
        x_hat = self.decoder_input(z)
        x_hat = x_hat.view(-1, 3, self.encoder_output_height, self.encoder_output_width)
        x_hat = self.decoder(x_hat)
        return x_hat


class SimpleVAE(ParentVAE):

    def __init__(self, in_shape: torch.Tensor, latent_nodes: int, **kwargs):
        super().__init__()

        in_chans = in_shape[0]
        height, width = in_shape[1], in_shape[2]
        self.encoder_output_height = tools.output_shape_conv2d(height, padding=1, kernel_size=3, stride=2, iterations=3)
        self._latent_nodes = latent_nodes

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.fc_mu = nn.Linear(64 * self.encoder_output_height**2, latent_nodes)
        self.fc_var = nn.Linear(64 * self.encoder_output_height**2, latent_nodes)

        self.decoder_input = nn.Linear(latent_nodes, 64 * self.encoder_output_height**2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1 if height == 64 else 0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.final_layer = nn.Sequential(

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, in_chans, kernel_size=3, padding=1),
            nn.Tanh())

    def decode(self, z: Tensor) -> Tensor:
        """
        Receives a variable sampled from q(z) and transforms it back to p(x|z)
        """
        x_hat = self.decoder_input(z)
        x_hat = x_hat.view(-1, 64, self.encoder_output_height, self.encoder_output_height)
        x_hat = self.decoder(x_hat)
        x_hat = self.final_layer(x_hat)
        return x_hat
