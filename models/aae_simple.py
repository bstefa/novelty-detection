import torch
import torch.nn as nn

from functools import reduce
from utils import tools
from utils.dtypes import *
from models.blocks import EncodingBlock, DecodingBlock


class ParentAAE(nn.Module):
    """
    Parent class for all AAE-style models, implements losses for the discriminator,
    generator, and along the reconstruction operations.
    """
    def __init__(self):
        super().__init__()

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
        batch_lt = self.encoder(batch_in)
        batch_rc = self.decoder(batch_lt)
        return loss_fn(batch_rc, batch_in)


class BaselineAAE(ParentAAE):
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.encoder = nn.Sequential(
            EncodingBlock(in_chans, 24),
            EncodingBlock(24, 48),
            EncodingBlock(48, 48, stride=2),
            EncodingBlock(48, 24),
            EncodingBlock(24, 16),
            EncodingBlock(16, 8, stride=2),
            nn.Conv2d(8, 3, kernel_size=5, padding=2),
            tools.Flatten(),
            nn.Linear(3 * self.encoder_output_height*self.encoder_output_width, latent_nodes))

        self.decoder = nn.Sequential(
            nn.Linear(latent_nodes, 3 * self.encoder_output_height*self.encoder_output_width),
            tools.Restructure((-1, 3, self.encoder_output_height, self.encoder_output_width)),
            DecodingBlock(3, 8),
            DecodingBlock(8, 16, stride=2, output_padding=(1, 0) if height == 248 else 1),
            DecodingBlock(16, 24),
            DecodingBlock(24, 48),
            DecodingBlock(48, 48, stride=2, output_padding=(1, 0) if height == 248 else 1),
            DecodingBlock(48, 24),
            nn.Conv2d(24, in_chans, kernel_size=5, padding=2),
            nn.Tanh())

        self.discriminator = nn.Sequential(
            tools.Flatten(),
            nn.Linear(latent_nodes, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid())


class SimpleAAE(ParentAAE):
    def __init__(self, in_shape: torch.Tensor, latent_nodes: int = 8):
        super().__init__()

        in_nodes = reduce(lambda x, y: x*y, in_shape)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = nn.Sequential(
            tools.Flatten(),
            nn.Linear(in_nodes, 1000),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(1000, latent_nodes))

        self.decoder = nn.Sequential(
            tools.Flatten(),
            nn.Linear(latent_nodes, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, in_nodes),
            tools.Restructure((-1, *in_shape)),
            nn.Sigmoid())

        # TODO: Consider initializing the weights explicitly here
        self.discriminator = nn.Sequential(
            tools.Flatten(),
            nn.Linear(latent_nodes, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Sigmoid())
