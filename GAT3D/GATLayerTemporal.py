import ipdb
import numpy as np
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb
from .unet import UNet


class GATLayerTemporal(nn.Module):
    # in_feature = out_feature (because here the feature is about the frame number)
    def __init__(self, in_features, out_features, alpha, conv=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(t.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.is_conv = conv
        self.conv_net = UNet(in_features, out_features)

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()  # 32, 5, 35, 35, 4
            h = h.permute(0, 4, 1, 2, 3)
        else:
            N, V, H, W = h.size()

        if self.is_conv:
            whs = []
            for i in range(H):
                whi = conv_net(h[:, i, :, :, :])
                whs.append(whi)
            Wh = t.cat(whs, axis=1)
        else:
            Wh = t.matmul(h, self.W)

        self.a = nn.Parameter(t.empty(size=(2 * H * W, 1))).to(
            self.W.device
        )  # added by Giulio
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.is_conv:
            Wh = self.conv_net(h)
        else:
            # ipdb.set_trace()
            # Wh = torch.matmul(h.double(), self.W.double())
            Wh = t.matmul(h, self.W)
        a_input = self.batch_prepare_attentional_mechanism_input(Wh)
        e = t.matmul(a_input, self.a)
        e = self.leakyrelu(e.squeeze(-1))
        attention = F.softmax(e, dim=-1)

        # Learnable Adjacency Matrix
        adj_mat = None
        self.B = nn.Parameter(t.zeros(V, V) + 1e-6).to(self.W.device)
        self.A = Variable(t.eye(V), requires_grad=False).to(self.W.device)
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = t.min(adj_mat)
        adj_mat_max = t.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(t.diag(t.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = t.sqrt(t.inverse(D))
        adj_mat_norm_d12 = t.matmul(t.matmul(D_12, adj_mat), D_12)

        Wh = Wh.view(N, V, H * W, T)  # Warning, T might be undefined
        attention = t.diag_embed(attention)
        Wh_ = []
        for i in range(V):
            at = t.zeros(N, H * W, T).to(self.W.device)  # added by Giulio
            for j in range(V):
                at += t.matmul(Wh[:, j, :, :], attention[:, i, j, :, :])
            Wh_.append(at)
        h_prime = t.stack((Wh_))
        h_prime = h_prime.permute(1, 2, 3, 0).contiguous().view(N, H * W, T, V)
        h_prime = t.matmul(h_prime, adj_mat_norm_d12).view(N, H, W, T, V)
        # return F.elu(h_prime)
        return t.sigmoid(F.elu(h_prime))

    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, H, W, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, M, 1, 1, 1)
        all_combinations_matrix = t.cat(
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
