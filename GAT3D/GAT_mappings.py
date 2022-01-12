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
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, x):
        return t.matmul(x, self.W)


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


class ConvMapping(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        pass

    def forward(self, x):
        pass
