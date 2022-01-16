from torch import nn
import torch.nn.functional as F
import torch as t

import math
from torch.autograd import Variable
import ipdb


dev = t.device("cuda:0") if t.cuda.is_available() else t.device("cpu")


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_vertices, alpha):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(t.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(t.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.B = nn.Parameter(t.zeros(n_vertices, n_vertices) + 1e-6)
        self.A = Variable(t.eye(n_vertices), requires_grad=False)

    def forward(self, h):
        if len(h.size()) == 4:
            N, C, T, V = h.size()
            h = h.permute(0, 3, 1, 2).contiguous().view(N, V, C * T)
        else:
            N, V, C = h.size()

        # GAT
        Wh = t.matmul(h, self.W)
        a_input = self.batch_prepare_attentional_mechanism_input(Wh)
        e = t.matmul(a_input, self.a)
        e = self.leakyrelu(e.squeeze(-1))
        attention = F.softmax(e, dim=-1)

        # Learnable Adjacency Matrix
        adj_mat = None
        self.A = self.A.cuda(h.get_device())
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = t.min(adj_mat)
        adj_mat_max = t.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(t.diag(t.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = t.sqrt(t.inverse(D))
        adj_mat_norm_d12 = t.matmul(t.matmul(D_12, adj_mat), D_12)

        # Updating the features of vertices
        attention = t.matmul(adj_mat_norm_d12, attention)
        h_prime = t.matmul(attention, Wh)

        return F.elu(h_prime)

    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, E = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)
        all_combinations_matrix = t.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
        )
        return all_combinations_matrix.view(B, M, M, 2 * E)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GATMultiHead(nn.Module):
    def __init__(self, nfeat, nhid, n_vertices, alpha, nheads):
        super().__init__()
        self.attentions = [
            GraphAttentionLayer(
                in_features=nfeat, out_features=nhid, n_vertices=n_vertices, alpha=alpha
            )
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, x):
        # N, C, T, V = x.size()
        N, C, V = x.size()
        x = t.cat([att(x) for att in self.attentions], dim=-1)
        """
        x = (
            x.view(N, V, C, T)
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(N, C, T, V)
        )
        """
        return x


class GraphAttentionLayer2D(nn.Module):
    def __init__(self, in_features, out_features, n_vertices, alpha):
        super(GraphAttentionLayer2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(t.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(t.empty(size=(2 * out_features, 1))).float()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.B = nn.Parameter(t.zeros(n_vertices, n_vertices) + 1e-6)
        self.A = Variable(t.eye(n_vertices), requires_grad=False)

    def forward(self, h):
        if len(h.size()) == 4:
            N, C, T, V = h.size()
            h = h.permute(0, 3, 1, 2)
        else:
            N, V, C = h.size()

        # GAT
        Wh = t.matmul(h, self.W)
        a_input = self.batch_prepare_attentional_mechanism_input(Wh).float()
        e = t.matmul(a_input, self.a)
        e = self.leakyrelu(e.squeeze(-1))
        attention = F.softmax(e, dim=-1)

        # Learnable Adjacency Matrix
        adj_mat = None
        self.A = self.A.cuda(h.get_device())
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = t.min(adj_mat)
        adj_mat_max = t.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(t.diag(t.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = t.sqrt(t.inverse(D))
        adj_mat_norm_d12 = t.matmul(t.matmul(D_12, adj_mat), D_12)

        # 2D Attention
        Wh = Wh.permute(0, 1, 3, 2)
        attention = t.diag_embed(attention)
        Wh_ = []
        for i in range(V):
            at = t.zeros(N, self.out_features, C).to(dev)
            for j in range(V):
                at += t.matmul(Wh[:, j, :, :].to(dev), attention[:, i, j, :, :].to(dev))
            Wh_.append(at)

        h_prime = t.stack((Wh_))
        h_prime = (
            h_prime.permute(1, 3, 2, 0).contiguous().view(N, C * self.out_features, V)
        )
        h_prime = t.matmul(h_prime, adj_mat_norm_d12).view(N, C, self.out_features, V)

        return F.elu(h_prime)

    def batch_prepare_attentional_mechanism_input(self, Wh):
        B, M, E, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, M, 1, 1)
        all_combinations_matrix = t.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
        )
        return all_combinations_matrix.view(B, M, M, E, 2 * T)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GATMultiHead2D(nn.Module):
    def __init__(self, nfeat, nhid, n_vertices, alpha, nheads):
        super().__init__()
        self.attentions = [
            GraphAttentionLayer2D(
                in_features=nfeat, out_features=nhid, n_vertices=n_vertices, alpha=alpha
            )
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, x):
        N, C, T, V = x.size()
        x = t.cat(tuple(att(x) for att in self.attentions), dim=2)
        return x


class BaselineModel2D(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        time_steps: int = 4,
        mapping_type="linear",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        self.hidden_layer = GATMultiHead2D(
            nfeat=time_steps,
            nhid=time_steps,
            n_vertices=n_vertices,
            alpha=0.2,
            nheads=1,
        )
        self.output_layer = GATMultiHead2D(
            nfeat=time_steps,
            nhid=time_steps,
            n_vertices=n_vertices,
            alpha=0.2,
            nheads=1,
        )

    def forward(self, x):
        B, H, W, T, V = x.shape
        x = x.reshape(B, H * W, T, V)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        x = x.view(B, H, W, T, V)
        return t.tanh(x)


class BaselineModel(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        time_steps: int = 4,
        mapping_type="linear",
    ):
        super().__init__()
        self.mapping_type = mapping_type
        n_features = time_steps * image_height * image_width
        self.hidden_layer = GATMultiHead(
            nfeat=n_features,
            nhid=n_features,
            n_vertices=n_vertices,
            alpha=0.2,
            nheads=1,
        )
        self.output_layer = GATMultiHead(
            nfeat=n_features,
            nhid=n_features,
            n_vertices=n_vertices,
            alpha=0.2,
            nheads=1,
        )

    def forward(self, x):
        B, H, W, T, V = x.shape
        x = x.reshape(B, H * W * T, V).permute(0, 2, 1)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        x = x.view(B, H, W, T, V)
        return t.tanh(x)
