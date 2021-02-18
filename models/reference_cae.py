
import torch.nn as nn
import torch.nn.functional as F

class ReferenceCAE(nn.Module):
    '''
    The lighting module helps enforce best practices
    by keeping your code modular and abstracting the
    'engineering code' or boilerplate that new model
    require.
    '''
    def __init__(
            self,
            params: dict
        ):
        super(ReferenceCAE, self).__init__()

        # Encoding layers
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, stride=2)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Decoding layers
        self.conv6_3 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv6_2 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv6_1 = nn.ConvTranspose2d(512, 512, 3, padding=1, stride=2, output_padding=1)
        self.conv7_3 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv7_2 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.conv7_1 = nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv8_3 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv8_2 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv8_1 = nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1)
        self.conv9_2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.conv9_1 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1)
        self.conv10_2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.conv10_1 = nn.ConvTranspose2d(64, 3, 3, padding=1, stride=2, output_padding=1)

        # Unpooling layers with bilinear interpolation
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Batch normalization layer
        self.bn3 = nn.BatchNorm2d(3)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)

    def encode(self, x):
        # print(torch.max(x), torch.min(x), x.size())
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn64(x)
        # x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn128(x)
        # x = self.pool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.bn256(x)
        # x = self.pool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.bn512(x)
        # x = self.pool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.bn512(x)
        # x = self.pool(x)
        z_latent = x
        # print(torch.max(x), torch.min(x), x.size())
        return z_latent

    def decode(self, z):
        # print(torch.max(z), torch.min(z), z.size())
        z = F.relu(self.conv6_3(z))
        z = F.relu(self.conv6_2(z))
        z = F.relu(self.conv6_1(z))
        z = self.bn512(z)
        # z = self.unpool(z)
        z = F.relu(self.conv7_3(z))
        z = F.relu(self.conv7_2(z))
        z = F.relu(self.conv7_1(z))
        z = self.bn256(z)
        # z = self.unpool(z)
        z = F.relu(self.conv8_3(z))
        z = F.relu(self.conv8_2(z))
        z = F.relu(self.conv8_1(z))
        z = self.bn128(z)
        # z = self.unpool(z)
        z = F.relu(self.conv9_2(z))
        z = F.relu(self.conv9_1(z))
        z = self.bn64(z)
        # z = self.unpool(z)
        z = F.relu(self.conv10_2(z))
        z = F.relu(self.conv10_1(z))
        z = self.bn3(z)
        # z = self.unpool(z)
        x_recons = z
        # print(torch.max(x_recons), torch.min(x_recons), x_recons.size())
        return x_recons

    def forward(self, x):
        # Simple encoding into latent representation
        # and decoding back to input space
        z_latent = self.encode(x)
        x_recons = self.decode(z_latent)
        return x_recons
