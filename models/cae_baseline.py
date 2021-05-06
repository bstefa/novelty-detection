import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.dtypes import *


class EncodingBlock(nn.Module):
    def __init__(
            self, in_chans: int, out_chans: int,
            kernel_size: int=5, padding: int=2, drop_prob: float=0.1, leak: float=0.1,
            **kwargs
    ):
        super(EncodingBlock, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, **kwargs)
        self.drop = nn.Dropout2d(p=drop_prob)
        self.activation = nn.LeakyReLU(negative_slope=leak)
        self.bn = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
            self, in_chans: int, out_chans: int,
            kernel_size: int=5, padding: int=2, drop_prob: float=0.1, leak: float=0.1,
            **kwargs
    ):
        super(DecodingBlock, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, **kwargs)
        self.drop = nn.Dropout2d(drop_prob)
        self.activation = nn.LeakyReLU(leak)
        self.bn = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = self.transconv(x)
        x = self.drop(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class BaselineCAE(nn.Module):
    def __init__(self, in_shape: Size):
        super().__init__()
        in_chans = in_shape[0]
        height, width = in_shape[1], in_shape[2]
        assert isinstance(in_chans, int) and isinstance(height, int), \
            f'in_chans must be of type int, got {type(in_chans)}, and {type(height)}'
        assert any((in_chans == nb for nb in [1, 3, 6])), \
            f'Input image must be greyscale (1), RGB/YUV (3), or 6-channel multispectral, got {in_chans} channels'
        # assert any((height == nb for nb in [28, 64, 256])), \
        #     f'Input image must have height == 28, 64, or 256, got {height}'

        # Encoding layers
        self.encoder = nn.Sequential(
            EncodingBlock(in_chans, 24),
            EncodingBlock(24, 48),
            EncodingBlock(48, 48, stride=2),
            EncodingBlock(48, 24),
            EncodingBlock(24, 16),
            EncodingBlock(16, 8, stride=2),
            nn.Conv2d(8, 3, kernel_size=5, padding=2),
        )

        # Decoding layers
        self.decoder = nn.Sequential(
            DecodingBlock(3, 8),
            DecodingBlock(8, 16, stride=2, output_padding=1),
            DecodingBlock(16, 24),
            DecodingBlock(24, 48),
            DecodingBlock(48, 48, stride=2, output_padding=1),
            DecodingBlock(48, 24),
            nn.Conv2d(24, in_chans, kernel_size=5, padding=2),
            nn.Tanh()  # Same size as input
        )

    def forward(self, x):
        # Simple encoding into latent representation and decoding back to input space
        x = self.encoder(x)
        x = self.decoder(x)
        return x
