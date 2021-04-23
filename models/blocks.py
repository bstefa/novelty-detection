from torch import nn


class EncodingBlock(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 5,
            padding: int = 2,
            drop_prob: float = 0.1,
            leak: float = 0.1,
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
            self,
            in_chans: int,
            out_chans: int,
            kernel_size: int = 5,
            padding: int = 2,
            drop_prob: float = 0.1,
            leak: float = 0.1,
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
