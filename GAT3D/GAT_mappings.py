import ipdb
import numpy as np
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

from .smaat_unet.SmaAt_UNet import SmaAt_UNet
from .autoencoder import Encoder, Decoder


class EncoderMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, x: t.Tensor):
        acc = []
        for i in range(x.shape[-1]):
            row = []
            for j in range(x.shape[1]):
                x_piece = x[:, j, :, :, i].unsqueeze(1)
                enc = self.encoder(x_piece).squeeze()
                row.append(enc)
            acc.append(t.stack(row, dim=1))
        result = t.stack(acc, dim=-1)
        result = result.view(x.shape[0], 6, 16, 16, 4)
        return result


class DecoderMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()

    def forward(self, x: t.Tensor):
        x = x.permute(0, 4, 1, 2, 3)
        acc = []
        for i in range(x.shape[-1]):
            row = []
            for j in range(x.shape[1]):
                x_piece = x[:, j, :, :, i].unsqueeze(1)
                enc = self.decoder(x_piece).squeeze()
                row.append(enc)
            acc.append(t.stack(row, dim=1))
        result = t.stack(acc, dim=-1)
        result = result.view(x.shape[0], 6, 16, 16, 4)
        # Eresult = result[:, :252, :, :, :].view(x.shape[0], 6, 12, 14, 4)
        return result


class LinearMapping(nn.Module):
    def __init__(self, mapping_type="temporal", in_features=4, out_features=4):
        super().__init__()
        self.W = nn.Parameter(t.empty(size=(in_features, out_features)))
        self.do = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        return self.do(t.matmul(x, self.W))


class SmaAt_UNetMapping(nn.Module):
    def __init__(
        self,
        mapping_type: str = "conv",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.unet = SmaAt_UNet(n_channels=4, n_classes=4)

    def forward(self, x):
        # ipdb.set_trace()
        old_x = x
        acc = []
        x = x.permute(1, 0, 4, 2, 3)
        for i in range(x.shape[0]):  # should be vertex dim
            acc.append(self.unet(x[i, :, :, :, :]))
        result = t.stack(acc)
        result = result.permute(1, 0, 3, 4, 2)
        return result + old_x


"""
class SmaAt_UNetMapping(nn.Module):
    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.conv_net = SmaAt_UNet(
            n_channels=in_features, n_classes=out_features
        )

    def forward(self, h):
        whs = []
        for i in range(h.shape[1]):
            # print(i, h[:, i, :, :, :].squeeze(1).permute(0, 3, 1, 2).size(), "111111111")
            whi = self.conv_net(
                h[:, i, :, :, :].squeeze(1).permute(0, 3, 1, 2)
            )
            whs.append(whi)
        Wh = t.stack(whs).permute(1, 0, 3, 4, 2)
        return Wh
"""


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
        chin: int,
        chout: int,
        first_kernel_dim: int,
        kernel_size: int,
        dropout=0.15,
        nonlinear=True,
        skip=True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            chin,
            chout,
            (first_kernel_dim, kernel_size, kernel_size),
            padding=(first_kernel_dim//2, kernel_size//2, kernel_size//2),
        )
        self.expansion = chout / chin
        self.bn = nn.BatchNorm3d(chin)
        self.do = nn.Dropout(dropout)
        self.skip = skip
        self.nonlinear = nonlinear

    def forward(self, x):
        # y_hat = self.bn(x)
        print(f"heyy {x=}")
        ipdb.set_trace()
        y_hat = self.do(self.conv(x))
        if self.nonlinear:
            y_hat = F.relu(y_hat)
        if self.skip:
            if self.expansion > 1:
                x_resized = x.repeat(1, 2, 1, 1, 1)
                y_hat = y_hat.clone() + x_resized
                y_hat = F.relu(y_hat)
            else:
                y_hat = y_hat.clone() + x[:, : y_hat.shape[1]]
                y_hat = F.relu(y_hat)

        return y_hat


class ConvMapping(nn.Module):
    def __init__(self, attention_type: str, skip=True):
        super().__init__()
        self.attention_type = attention_type
        groups = 4 if attention_type == "spatial" else 6
        first_kernel_dim = 6 if attention_type == "spatial" else 4
        self.skip = skip
        self.net = nn.Sequential(
            # ConvBlock2D(6, 6, 0.15, False)
            # ConvBlock3D(groups, groups, 2, 3, 0.10, True),
            ConvBlock3D(groups, groups * 2, 2, 3, 0.10, True),
            ConvBlock3D(groups * 2, groups * 4, 2, 3, 0.10, True),
            # ConvBlock3D(groups * 4, groups * 8, 2, 3, 0.10, True),
            # ConvBlock3D(groups * 8, groups * 16, 2, 3, 0.10, True),
            # ConvBlock3D(groups * 16, groups * 8, 2, 3, 0.10, True),
            # ConvBlock3D(groups * 8, groups * 4, 2, 3, 0.10, True),
            ConvBlock3D(groups * 4, groups * 2, 2, 3, 0.10, True),
            ConvBlock3D(groups * 2, groups, 2, 3, 0.10, True),
            # ConvBlock3D(6, 6, 5, 0.15, False),
            # ConvBlock3D(6, 6, 3, 0.15, False),
        )

    def resape_input(self, x):
        pass

    def forward(self, x):
        old_x = x
        if self.attention_type == "spatial":
            x = x.permute(0, 4, 1, 2, 3)
        else:
            x = x.permute(0, 1, 4, 2, 3)

        x = self.net(x)
        if self.attention_type == "spatial":
            y_hat = x.permute(0, 2, 3, 4, 1)
        else:
            y_hat = x.permute(0, 1, 3, 4, 2)
        return y_hat  # + old_x if self.skip else y_hat
