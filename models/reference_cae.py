import torch.nn as nn


class EncodingBlock(nn.Module):
    def __init__(
            self, in_chans: int, out_chans: int,
            kernel_size: int=5, padding: int=2, drop_prob: float=0.1, leak: float=0.1,
            **kwargs
    ):
        super(EncodingBlock, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, **kwargs)
        self.activation = nn.LeakyReLU(negative_slope=leak)
        self.drop = nn.Dropout2d(p=drop_prob)
        self.bn = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.drop(x)
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
        self.activation = nn.LeakyReLU(leak)
        self.drop = nn.Dropout2d(drop_prob)
        self.bn = nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.bn(x)
        return x


class ReferenceCAE(nn.Module):
    def __init__(self, in_shape: int):
        super(ReferenceCAE, self).__init__()
        c = in_shape[0]

        # Encoding layers
        self.encoder = nn.Sequential(
            EncodingBlock(c, 24),
            EncodingBlock(24, 48),
            EncodingBlock(48, 48, stride=2),
            EncodingBlock(48, 24),
            EncodingBlock(24, 16),
            EncodingBlock(16, 8, stride=2),
            nn.Conv2d(8, c, kernel_size=5, padding=2),
        )

        # Decoding layers
        self.decoder = nn.Sequential(
            DecodingBlock(c, 8),
            DecodingBlock(8, 16, stride=2, output_padding=1),
            DecodingBlock(16, 24),
            DecodingBlock(24, 48),
            DecodingBlock(48, 48, stride=2, output_padding=1),
            DecodingBlock(48, 24),
            nn.Conv2d(24, c, kernel_size=5, padding=2),
            nn.Tanh()  # Same size as input
        )

    def forward(self, x):
        # Simple encoding into latent representation and decoding back to input space
        x = self.encoder(x)
        x = self.decoder(x)
        return x
