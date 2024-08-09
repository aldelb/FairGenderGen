""" Parts of the U-Net model """
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import constants.constants as constants
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    

class Conv(pl.LightningModule):

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same", bias=True),
            nn.Dropout(constants.dropout),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(pl.LightningModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel, padding="same", bias=False),
            nn.Dropout(constants.dropout),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel, padding="same", bias=False),
            nn.Dropout(constants.dropout),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownDiscr(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            Conv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class DownSimple(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            Conv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Down(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class UpSimple(pl.LightningModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
            self.conv = Conv(in_channels, out_channels, kernel)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel, stride=2)
            self.conv = Conv(in_channels, out_channels,  kernel)

    def forward(self, x1, x2 = None):
        if(x2 == None):
            x1 = self.up(x1)
            return self.conv(x1)
        else:
            x1 = self.up(x1)
            diff = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, [diff // 2, diff - diff// 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
    
class Up(pl.LightningModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel, in_channels)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,  kernel)

    def forward(self, x1, x2 = None):
        if(x2 == None):
            x1 = self.up(x1)
            return self.conv(x1)
        else:
            x1 = self.up(x1)
            diff = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, [diff // 2, diff - diff// 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
    
class DoubleConvLayerNorm(pl.LightningModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel, lenght, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel, padding="same", bias=False),
            nn.Dropout(constants.dropout),
            nn.LayerNorm([mid_channels, lenght]),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel, padding="same", bias=False),
            nn.Dropout(constants.dropout),
            nn.LayerNorm([out_channels, lenght]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownLayerNorm(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel, lenght):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConvLayerNorm(in_channels, out_channels, kernel, floor(lenght/2))
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class UpLayerNorm(pl.LightningModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel, lenght, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConvLayerNorm(in_channels, out_channels, kernel, round(lenght*2))
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel, stride=2)
            self.conv = DoubleConvLayerNorm(in_channels, out_channels, kernel, round(lenght*2))

    def forward(self, x1, x2 = None):
        if(x2 == None):
            x1 = self.up(x1)
            return self.conv(x1)
        else:
            x1 = self.up(x1)
            diff = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, [diff // 2, diff - diff// 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


class OutConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same")

    def forward(self, x):
        return self.conv(x)