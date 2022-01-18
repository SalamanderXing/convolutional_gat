import ipdb
import numpy as np
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

from .smaat_unet.SmaAt_UNet import SmaAt_UNet


class LinearMapping(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(t.empty(size=(in_features, out_features)))
        self.do = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        return self.do(t.matmul(x, self.W))


class SmaAt_UNetMapping(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv_net = SmaAt_UNet(n_channels=in_features, n_classes=out_features)

    def forward(self, h):
        whs = []
        for i in range(h.shape[1]):
            # print(i, h[:, i, :, :, :].squeeze(1).permute(0, 3, 1, 2).size(), "111111111")
            whi = self.conv_net(h[:, i, :, :, :].squeeze(1).permute(0, 3, 1, 2))
            whs.append(whi)
        Wh = t.stack(whs).permute(1, 0, 3, 4, 2)
        return Wh


class ConvBlock2D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        dropout=0.15,
        nonlinear=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            padding="same",
            groups=in_features,
        )
        self.do = nn.Dropout(dropout)
        self.nonlinear = nonlinear

    def forward(self, x):
        y_hat = self.do(
            t.stack(tuple(self.conv(x[:, i]) for i in range(x.shape[1])))
        ).permute(1, 0, 2, 3, 4)
        if self.nonlinear:
            y_hat = F.relu(y_hat)
        return y_hat


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        dropout=0.15,
        nonlinear=True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            4,
            4,
            (3, kernel_size, kernel_size),
            padding="same",
            groups=4,
        )
        self.do = nn.Dropout(dropout)
        self.nonlinear = nonlinear

    def forward(self, x):
        y_hat = self.do(self.conv(x))
        if self.nonlinear:
            y_hat = F.relu(y_hat)
        return y_hat


class ConvMapping(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            # ConvBlock2D(6, 6, 0.15, False)
            ConvBlock3D(6, 6, 3, 0.15, True),
            # ConvBlock3D(6, 6, 5, 0.15, False),
            # ConvBlock3D(6, 6, 3, 0.15, False),
        )

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.net(x)
        y_hat = x.permute(0, 2, 3, 4, 1)
        return y_hat
