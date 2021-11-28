import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .unet import UNet


class GATLayerSpatial(nn.Module):
    def __init__(self, in_features, out_features, alpha, conv=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # [4, 4]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # [8, 1]
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.is_conv = conv
        if self.is_conv:
            self.conv = UNet(in_features, out_features)

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()
            h = h.permute(0, 4, 1, 2, 3)
        else:
            N, V, H, W = h.size()

        if self.is_conv:
            Wh = self.conv(h)
        else:
            Wh = torch.matmul(h, self.W)
        a_input = self.batch_prepare_attentional_mechanism_input(Wh)
        e = torch.matmul(a_input, self.a)
        e = self.leakyrelu(e.squeeze(-1))
        attention = F.softmax(e, dim=-1)

        # Learnable Adjacency Matrix
        adj_mat = None
        self.B = nn.Parameter(torch.zeros(V, V) + 1e-6)
        self.A = Variable(torch.eye(V), requires_grad=False)
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.matmul(torch.matmul(D_12, adj_mat), D_12)

        Wh_ = []
        for i in range(V):
            at = torch.zeros(N, H, W, self.out_features)
            for j in range(V):
                at += Wh[:, j, :, :, :] * attention[:, i, j, :, :].unsqueeze(3).repeat(
                    1, 1, 1, self.out_features
                )
            Wh_.append(at)

        h_prime = torch.stack((Wh_))
        h_prime = torch.stack((Wh_))
        h_prime = (
            h_prime.permute(1, 3, 4, 2, 0)
            .contiguous()
            .view(N, H, W * self.out_features, V)
        )
        h_prime = torch.matmul(h_prime, adj_mat_norm_d12).view(
            N, H, W, self.out_features, V
        )
        return F.elu(h_prime)

    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, H, W, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, M, 1, 1, 1)
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
        )
        return all_combinations_matrix.view(B, M, M, H, W, 2 * T)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
