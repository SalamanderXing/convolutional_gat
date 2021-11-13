import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GATLayerTemporal(nn.Module):
    # in_feature = out_feature (because here the feature is about the frame number)
    def __init__(self, in_features, out_features, alpha):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()
            h = h.permute(0, 4, 1, 2, 3)
        else:
            N, V, H, W = h.size()

        self.a = nn.Parameter(torch.empty(size=(2 * H * W, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

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

        Wh = Wh.view(N, V, H * W, T)
        attention = torch.diag_embed(attention)
        Wh_ = []
        for i in range(V):
            at = torch.zeros(N, H * W, T)
            for j in range(V):
                at += torch.matmul(Wh[:, j, :, :], attention[:, i, j, :, :])
            Wh_.append(at)
        h_prime = torch.stack((Wh_))
        h_prime = h_prime.permute(1, 2, 3, 0).contiguous().view(N, H * W, T, V)
        h_prime = torch.matmul(h_prime, adj_mat_norm_d12).view(N, H, W, T, V)
        return F.elu(h_prime)

    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, H, W, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, M, 1, 1, 1)
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-2
        )
        return all_combinations_matrix.view(B, M, M, T, 2 * H * W)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
