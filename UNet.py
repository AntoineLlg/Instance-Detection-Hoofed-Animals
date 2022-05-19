import torch
import torch.nn as nn
from utils import TorchMinMaxScaler


def relu_conv_down(in_channels, out_channels):
    return nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    )


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dconv_down1 = relu_conv_down(in_channels, 32)
        self.dconv_down2 = relu_conv_down(32, 64)
        self.dconv_down3 = relu_conv_down(64, 128)
        self.dconv_down4 = relu_conv_down(128, 256)

        self.simpleconv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.simpleconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.simpleconv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_channels, 1)
        self.MinMaxScaler = TorchMinMaxScaler()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.simpleconv1(conv1)

        conv2 = self.dconv_down2(x)
        x = self.simpleconv2(conv2)

        conv3 = self.dconv_down3(x)
        x = self.simpleconv3(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        out = self.MinMaxScaler.fit_transform(x)
        return out