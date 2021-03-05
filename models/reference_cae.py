import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceCAE(nn.Module):
    def __init__(
            self,
            params: dict
    ):
        super(ReferenceCAE, self).__init__()

        # Encoding layers
        self.conv1_1 = nn.Conv2d(3, 24, 5, padding=2)
        self.conv1_1_bn = nn.BatchNorm2d(24)
        self.conv1_2 = nn.Conv2d(24, 48, 5, padding=2)
        self.conv1_2_bn = nn.BatchNorm2d(48)
        self.conv2_1 = nn.Conv2d(48, 48, 5, padding=2, stride=2)
        self.conv2_1_bn = nn.BatchNorm2d(48)
        self.conv2_2 = nn.Conv2d(48, 24, 5, padding=2)
        self.conv2_2_bn = nn.BatchNorm2d(24)
        self.conv2_3 = nn.Conv2d(24, 16, 5, padding=2)
        self.conv2_3_bn = nn.BatchNorm2d(16)
        self.conv3_1 = nn.Conv2d(16, 8, 5, padding=2, stride=2)
        self.conv3_1_bn = nn.BatchNorm2d(8)
        self.conv3_2 = nn.Conv2d(8, 3, 5, padding=2)

        # Decoding layers
        self.transconv1_1 = nn.ConvTranspose2d(3, 8, 5, padding=2)
        self.transconv1_1_bn = nn.BatchNorm2d(8)
        self.transconv1_2 = nn.ConvTranspose2d(8, 16, 5, padding=2, stride=2, output_padding=1)
        self.transconv1_2_bn = nn.BatchNorm2d(16)
        self.transconv2_1 = nn.ConvTranspose2d(16, 24, 5, padding=2)
        self.transconv2_1_bn = nn.BatchNorm2d(24)
        self.transconv2_2 = nn.ConvTranspose2d(24, 48, 5, padding=2)
        self.transconv2_2_bn = nn.BatchNorm2d(48)
        self.transconv2_3 = nn.ConvTranspose2d(48, 48, 5, padding=2, stride=2, output_padding=1)
        self.transconv2_3_bn = nn.BatchNorm2d(48)
        self.transconv3_1 = nn.ConvTranspose2d(48, 24, 5, padding=2)
        self.transconv3_1_bn = nn.BatchNorm2d(24)
        self.transconv3_2 = nn.ConvTranspose2d(24, 3, 5, padding=2)

    def encode(self, x):
        # print('e-in', x.size())
        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.1)
        x = self.conv1_1_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.1)
        x = self.conv1_2_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.1)
        x = self.conv2_1_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.1)
        x = self.conv2_2_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.conv2_3(x), negative_slope=0.1)
        x = self.conv2_3_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.conv3_1(x), negative_slope=0.1)
        x = self.conv3_1_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.conv3_2(x), negative_slope=0.1)
        # print('e-out', torch.max(x), torch.min(x), x.size())
        return x

    def decode(self, x):
        # print('d-in', torch.max(x), torch.min(x), x.size())
        x = F.leaky_relu(self.transconv1_1(x), negative_slope=0.1)
        x = self.transconv1_1_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.transconv1_2(x), negative_slope=0.1)
        x = self.transconv1_2_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.transconv2_1(x), negative_slope=0.1)
        x = self.transconv2_1_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.transconv2_2(x), negative_slope=0.1)
        x = self.transconv2_2_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.transconv2_3(x), negative_slope=0.1)
        x = self.transconv2_3_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.transconv3_1(x), negative_slope=0.1)
        x = self.transconv3_1_bn(x)
        # print(x.size())
        x = F.leaky_relu(self.transconv3_2(x), negative_slope=0.1)
        # print('d-out', torch.max(x), torch.min(x), x.size())
        return x

    def forward(self, x):
        # Simple encoding into latent representation and decoding back to input space
        x = self.encode(x)
        x = self.decode(x)
        return x
