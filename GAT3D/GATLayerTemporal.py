import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class GATLayerTemporal(nn.Module):

    def __init__(self, image_size, in_features, out_features, alpha, concat=True):
        super(GATLayerTemporal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.img_height, self.img_width = image_size

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(
            size=(2 * self.img_height * self.img_width, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        nodes, height, width, frames = input.shape
        Wh = torch.matmul(input, self.W)

        # Attention 2D
        a_input = torch.cat([Wh.repeat(1, 1, nodes, 1).view(nodes*nodes, height, width, self.out_features),
                            Wh.repeat(nodes, 1, 1, 1)], dim=1).view(nodes, nodes, self.out_features, 2 * height * width)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)
        adj = (adj > 0).view(nodes, nodes, 1)
        attention = torch.where(adj, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        Wh_ = []
        for i in range(nodes):
            at = torch.zeros(height, width, self.out_features)
            for j in range(nodes):
                # print('====', Wh[j, :, :, :].shape, attention[i, j, :].unsqueeze(0).unsqueeze(0).repeat((height, width, 1)).shape)
                at += Wh[j, :, :, :] * attention[i, j,
                                                 :].unsqueeze(0).unsqueeze(0).repeat((height, width, 1))
            Wh_.append(at)

        h_prime = torch.stack((Wh_))

        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime
