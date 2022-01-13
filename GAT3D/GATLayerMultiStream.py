import ipdb
import numpy as np
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .GAT_mappings import LinearMapping, SmaAt_UNetMapping, ConvMapping


class GATLayerMultiStream(nn.Module):
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
        
        self.a_temporal = nn.Parameter(t.empty(size=(2 * image_height * image_width, 1)))
        nn.init.xavier_uniform_(self.a_temporal.data, gain=1.414)

        self.a_spatial = nn.Parameter(t.empty(size=(2 * out_features, 1)))  # [8, 1]
        nn.init.xavier_uniform_(self.a_spatial.data, gain=1.414)

        self.W = nn.Parameter(t.empty(size=(in_features, out_features)))  # [4, 4]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.B = nn.Parameter(t.zeros(n_vertices, n_vertices) + 1e-6)
        self.A = Variable(t.eye(n_vertices), requires_grad=False)

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()
            h = h.permute(0, 4, 1, 2, 3)
        else:
            N, V, H, W = h.size()

        if self.A.device != self.B.device:
            self.A = self.A.to(self.B.device)

        Wh = self.mapping(h)

        # Spatial Attention
        a_input_spatial = self.batch_prepare_attentional_mechanism_input(Wh)
        e_spatial = t.matmul(a_input_spatial, self.a_spatial)
        e_spatial = self.leakyrelu(e_spatial.squeeze(-1))
        attention_spatial = F.softmax(e_spatial, dim=-1)

        # Learnable Adjacency Matrix
        adj_mat = None
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = t.min(adj_mat)
        adj_mat_max = t.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(t.diag(t.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = t.sqrt(t.inverse(D))
        adj_mat_norm_d12 = t.matmul(t.matmul(D_12, adj_mat), D_12)

        Wh_ = []
        for i in range(V):
            at = t.zeros(N, H, W, self.out_features).to(self.W.device)
            for j in range(V):
                at += Wh[:, j, :, :, :] * attention_spatial[:, i, j, :, :].unsqueeze(3).repeat(
                    1, 1, 1, self.out_features
                )
            Wh_.append(at)

        h_prime_spatial = t.stack((Wh_))
        h_prime_spatial = t.stack((Wh_))
        h_prime_spatial = (
            h_prime_spatial.permute(1, 3, 4, 2, 0)
            .contiguous()
            .view(N, H, W * self.out_features, V)
        )
        h_prime_spatial = t.matmul(h_prime_spatial, adj_mat_norm_d12).view(
            N, H, W, self.out_features, V
        )

        # Temporal Attention
        a_input_temporal = self.batch_prepare_attentional_mechanism_input(Wh)
        e_temporal = t.matmul(a_input_temporal, self.a_temporal)
        e_temporal = self.leakyrelu(e_temporal.squeeze(-1))
        attention_temporal = F.softmax(e_temporal, dim=-1)

        Wh = Wh.view(N, V, H * W, T)  # Warning, T might be undefined
        attention_temporal = t.diag_embed(attention_temporal)
        Wh_ = []
        for i in range(V):
            at = t.zeros(N, H * W, T).to(self.B.device)  # added by Giulio
            for j in range(V):
                at += t.matmul(Wh[:, j, :, :], attention_temporal[:, i, j, :, :])
            Wh_.append(at)
        h_prime_temporal = t.stack((Wh_))
        h_prime_temporal = h_prime_temporal.permute(1, 2, 3, 0).contiguous().view(N, H * W, T, V)
        h_prime_temporal = t.matmul(h_prime_temporal, adj_mat_norm_d12).view(N, H, W, T, V)

        print("********* ensemble shape ********", h_prime_temporal.size())


        return F.elu(h_prime)

    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, H, W, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, M, 1, 1, 1)
        all_combinations_matrix = t.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
        )
        return all_combinations_matrix.view(B, M, M, H, W, 2 * T)