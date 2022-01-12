import ipdb
import numpy as np
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb
from .GAT_mappings import LinearMapping, SmaAt_UNetMapping, ConvMapping

# from .unet import UNet


class GATLayerTemporal(nn.Module):
    # in_feature = out_feature (because here the feature is about the frame number)
    def __init__(
        self,
        in_features,
        out_features,
        alpha,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        mapping_type="conv",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.mapping_type = mapping_type
        if self.mapping_type == "linear":
            mappingClass = LinearMapping
        elif self.mapping_type == "smaat_unet":
            mappingClass = SmaAt_UNetMapping
        elif self.mapping_type == "conv":
            mappingClass = ConvMapping
        else:
            raise TypeError(f"Mapping type not supported: {self.type}")
        self.mapping = mappingClass(in_features, out_features)
        self.a = nn.Parameter(t.empty(size=(2 * image_height * image_width, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.B = nn.Parameter(t.zeros(n_vertices, n_vertices) + 1e-6)
        self.A = Variable(t.eye(n_vertices), requires_grad=False)

        # self.conv_net = UNet(in_features, out_features)

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()  # 32, 5, 35, 35, 4
            h = h.permute(0, 4, 1, 2, 3)
            # print("five", H)
        else:
            # print("not five")
            N, V, H, W = h.size()

        if self.A.device != self.B.device:
            self.A = self.A.to(self.B.device)
        # print("=======", h.size())
        Wh = self.mapping(h)

        # print("***********", Wh.size())

        # if self.is_conv:
        #     whs = []
        #     print(h.size())
        #     for i in range(h.size()[2]):
        #         print(h[:, :, i, :, :].size(), h[:, :, i, :, :].squeeze().permute(0, 3, 1, 2).size())
        #         whi = self.conv_net(h[:, :, i, :, :].squeeze().permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        #         whs.append(whi)
        #     Wh = t.cat(whs, axis=1)

        # if self.is_conv:
        #     Wh = self.conv_net(h)
        # else:
        #     Wh = t.matmul(h, self.W)

        a_input = self.batch_prepare_attentional_mechanism_input(Wh)
        e = t.matmul(a_input, self.a)
        e = self.leakyrelu(e.squeeze(-1))
        attention = F.softmax(e, dim=-1)

        # Learnable Adjacency Matrix
        adj_mat = None
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
            at = t.zeros(N, H * W, T).to(self.B.device)  # added by Giulio
            for j in range(V):
                at += t.matmul(Wh[:, j, :, :], attention[:, i, j, :, :])
            Wh_.append(at)
        h_prime = t.stack((Wh_))
        h_prime = h_prime.permute(1, 2, 3, 0).contiguous().view(N, H * W, T, V)
        h_prime = t.matmul(h_prime, adj_mat_norm_d12).view(N, H, W, T, V)
        # return F.elu(h_prime)
        return F.elu(h_prime)

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
