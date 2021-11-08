import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class GATLayerSpatial(nn.Module):

    def __init__(self, in_features, out_features, alpha, concat=True, ):
        super(GATLayerSpatial, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        nodes, height, width, frames = input.shape
        Wh = torch.matmul(input, self.W)
        a_input = torch.cat([Wh.repeat(1, 1, nodes, 1).view(nodes*nodes, height, width, self.out_features),
                            Wh.repeat(nodes, 1, 1, 1)], dim=2).view(nodes, nodes, height, width, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(4))

        zero_vec = -9e15*torch.ones_like(e)
        adj = (adj > 0).view(nodes, nodes, 1, 1)
        attention = torch.where(adj, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        Wh_ = []
        for i in range(nodes):
            at = torch.zeros(height, width, self.out_features)
            for j in range(nodes):
                at += Wh[j, :, :, :] * attention[i, j, :,
                                                 :].unsqueeze(2).repeat(1, 1, self.out_features)
            Wh_.append(at)

        h_prime = torch.stack((Wh_))

        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime
