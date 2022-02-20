import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .unet import UNet
from .GAT_mappings import LinearMapping, SmaAt_UNetMapping, ConvMapping
import ipdb


class Model(nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_vertices: int,
        attention_type: str = "multi_istream",
        time_steps: int = 4,
        mapping_type="linear",
        n_heads_per_layer=(4, 4),
    ):

        super().__init__()
        self.mapping_type = mapping_type
        self.layers = nn.Sequential(
            *[
                GATMultiHead3D(
                    nfeat=time_steps,
                    nhid=time_steps,
                    alpha=0.2,
                    nheads=n_heads_per_layer[i],
                    attention_type=attention_type,
                    mapping_type=mapping_type,
                    image_height=image_height,
                    image_width=image_width,
                    n_vertices=n_vertices,
                )
                for i in range(len(n_heads_per_layer))
            ]
        )

    def forward(self, x):
        return self.layers(x)


class GATMultiHead3D(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        alpha,
        nheads,
        *,
        attention_type: str = "temporal",
        image_width: int,
        image_height: int,
        n_vertices: int,
        mapping_type="conv",
    ):
        super().__init__()
        gatClass = {
            "temporal": GatTemporal,
            "spatial": GatSpatial,
            "multi_stream": GatMultistream,
        }[attention_type]
        self.attentions = [
            gatClass(
                in_features=nfeat,
                out_features=nhid,
                alpha=alpha,
                mapping_type=mapping_type,
                image_width=image_width,
                image_height=image_height,
                n_vertices=n_vertices,
            )
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, x):
        # N, H, W, T, V = x.size()
        # x = torch.cat(tuple(att(x) for att in self.attentions), dim=2)
        # ipdb.set_trace()
        # print("inputtttttt", x.size())
        x = t.mean(t.stack([att(x) for att in self.attentions]), dim=0)
        # print("outputtttttt", x.size())
        return x


def temporal_batch_prepare_attentional_mechanism_input(Wh):
    B, V, H, W, T = Wh.shape
    Wh_repeated_in_chunks = Wh.repeat_interleave(V, dim=1)
    Wh_repeated_alternating = Wh.repeat(1, V, 1, 1, 1)
    all_combinations_matrix = t.cat(
        [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-2
    )
    return all_combinations_matrix.view(B, V, V, T, 2 * H * W)


def spatial_batch_prepare_attentional_mechanism_input(Wh):
    B, V, H, W, T = Wh.shape
    Wh_repeated_in_chunks = Wh.repeat_interleave(V, dim=1)
    Wh_repeated_alternating = Wh.repeat(1, V, 1, 1, 1)
    all_combinations_matrix = t.cat(
        [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
    )
    return all_combinations_matrix.view(B, V, V, H, W, 2 * T)


def compute_attention(attention_type: str, Wh, a, leakyrelu):
    a_input = (
        spatial_batch_prepare_attentional_mechanism_input(Wh)
        if attention_type == "spatial"
        else temporal_batch_prepare_attentional_mechanism_input(Wh)
    )
    e = t.matmul(a_input, a)
    e = leakyrelu(e.squeeze(-1))
    return F.softmax(e, dim=-1)


def temporal_forward(self, N, V, T, H, W, Wh, adj_mat_norm_d12):
    attention = self.compute_attention("temporal", Wh)
    attention = t.diag_embed(attention)
    Wh = Wh.view(N, V, H * W, T)  # Warning, T might be undefined
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


class GatMultistream(nn.Module):
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
        self.temporal_gat = GatTemporal(
            in_features,
            out_features,
            alpha,
            image_width=image_width,
            image_height=image_height,
            n_vertices=n_vertices,
            mapping_type=mapping_type,
        )
        self.spatial_gat = GatTemporal(
            in_features,
            out_features,
            alpha,
            image_width=image_width,
            image_height=image_height,
            n_vertices=n_vertices,
            mapping_type=mapping_type,
        )

    def forward(self, x):
        x_temporal = self.temporal_gat(x)
        x_spatial = self.spatial_gat(x)
        result = t.mean(t.stack((x_temporal, x_spatial), dim=-1), dim=-1)
        return result


class GatSpatial(nn.Module):
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
        self.W = nn.Parameter(
            t.empty(size=(in_features, out_features))
        )  # [4, 4]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_spatial = nn.Parameter(
            t.empty(size=(2 * out_features, 1))
        )  # [8, 1]
        nn.init.xavier_uniform_(self.a_spatial.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.is_conv = conv
        self.B = nn.Parameter(t.zeros(n_vertices, n_vertices) + 1e-6)
        self.A = Variable(t.eye(n_vertices), requires_grad=False)

        self.mapping_type = mapping_type
        if self.mapping_type == "linear":
            mappingClass = LinearMapping
        elif self.mapping_type == "smaat_unet":
            mappingClass = SmaAt_UNetMapping
        elif self.mapping_type == "conv":
            mappingClass = ConvMapping
        else:
            raise TypeError(f"Mapping type not supported: {self.mapping_type}")
        self.spatial_mapping = mappingClass(
            "temporal"
        )  # not a bug that is says temporal

    def spatial_forward(self, N, V, T, H, W, Wh, adj_mat_norm_d12):
        attention = compute_attention(
            "spatial", Wh, self.a_spatial, self.leakyrelu
        )
        Wh_ = []
        for i in range(V):
            at = t.zeros(N, H, W, self.out_features).to(self.W.device)
            for j in range(V):
                at += Wh[:, i, :, :, :] * attention[:, i, j, :, :].unsqueeze(
                    3
                )  # .repeat(1, 1, 1, self.out_features) # was this useful??
            Wh_.append(at)
        h_prime = t.stack((Wh_))
        h_prime = (
            h_prime.permute(1, 3, 4, 2, 0).contiguous()
            # .view(N, H, W * self.out_features, V) was this useful??
        )
        # ipdb.set_trace()
        h_prime = t.matmul(h_prime, adj_mat_norm_d12).permute(0, 1, 3, 2, 4)
        # .view( # this was the bug!
        #    N, H, W, self.out_features, V
        # )
        # ipdb.set_trace()
        return F.elu(h_prime)

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()
            h = h.permute(0, 4, 1, 2, 3)
        else:
            N, V, H, W = h.size()

        if self.A.device != self.B.device:
            self.A = self.A.to(self.B.device)

        Wh = self.spatial_mapping(h)

        # Learnable Adjacency Matrix
        adj_mat = None
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = t.min(adj_mat)
        adj_mat_max = t.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(t.diag(t.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = t.sqrt(t.inverse(D))
        adj_mat_norm_d12 = t.matmul(t.matmul(D_12, adj_mat), D_12)
        h_prime = self.spatial_forward(N, V, T, H, W, Wh, adj_mat_norm_d12)
        return h_prime


class GatTemporal(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        alpha,
        *,
        attention_type="multi_istream",
        image_width: int,
        image_height: int,
        n_vertices: int,
        mapping_type="conv",
    ):
        super().__init__()
        self.attention_type = attention_type
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(
            t.empty(size=(in_features, out_features))
        )  # [4, 4]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_temporal = nn.Parameter(
            t.empty(size=(2 * image_height * image_width, 1))
        )
        nn.init.xavier_uniform_(self.a_temporal.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.is_conv = conv
        self.B = nn.Parameter(t.zeros(n_vertices, n_vertices) + 1e-6)
        self.A = Variable(t.eye(n_vertices), requires_grad=False)

        self.mapping_type = mapping_type
        if self.mapping_type == "linear":
            mappingClass = LinearMapping
        elif self.mapping_type == "smaat_unet":
            mappingClass = SmaAt_UNetMapping
        elif self.mapping_type == "conv":
            mappingClass = ConvMapping
        else:
            raise TypeError(f"Mapping type not supported: {self.mapping_type}")
        self.temporal_mapping = mappingClass("spatial")

    def temporal_forward(self, N, V, T, H, W, Wh, adj_mat_norm_d12):
        attention = compute_attention(
            "temporal", Wh, self.a_temporal, self.leakyrelu
        )
        attention = t.diag_embed(attention)
        Wh = Wh.view(N, V, H * W, T)  # Warning, T might be undefined
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

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()
            h = h.permute(0, 4, 1, 2, 3)
        else:
            N, V, H, W = h.size()

        if self.A.device != self.B.device:
            self.A = self.A.to(self.B.device)

        Wh = self.temporal_mapping(h)

        # Learnable Adjacency Matrix
        adj_mat = None
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = t.min(adj_mat)
        adj_mat_max = t.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(t.diag(t.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = t.sqrt(t.inverse(D))
        adj_mat_norm_d12 = t.matmul(t.matmul(D_12, adj_mat), D_12)
        h_prime = self.temporal_forward(N, V, T, H, W, Wh, adj_mat_norm_d12)
        return h_prime


"""
class BaseGat(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        alpha,
        *,
        attention_type="multi_istream",
        image_width: int,
        image_height: int,
        n_vertices: int,
        mapping_type="linear",
    ):
        super().__init__()
        self.attention_type = attention_type
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(
            t.empty(size=(in_features, out_features))
        )  # [4, 4]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if self.attention_type in ("spatial", "multi_stream"):
            self.a_spatial = nn.Parameter(
                t.empty(size=(2 * out_features, 1))
            )  # [8, 1]
            nn.init.xavier_uniform_(self.a_spatial.data, gain=1.414)
        if self.attention_type in ("temporal", "multi_stream"):
            self.a_temporal = nn.Parameter(
                t.empty(size=(2 * image_height * image_width, 1))
            )
            nn.init.xavier_uniform_(self.a_temporal.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # self.is_conv = conv
        self.B = nn.Parameter(t.zeros(n_vertices, n_vertices) + 1e-6)
        self.A = Variable(t.eye(n_vertices), requires_grad=False)

        self.mapping_type = mapping_type
        if self.mapping_type == "linear":
            mappingClass = LinearMapping
        elif self.mapping_type == "smaat_unet":
            mappingClass = SmaAt_UNetMapping
        elif self.mapping_type == "conv":
            mappingClass = ConvMapping
        else:
            raise TypeError(f"Mapping type not supported: {self.mapping_type}")
        self.temporal_mapping = mappingClass("temporal")
        self.spatial_mapping = mappingClass("spatial")

    def temporal_batch_prepare_attentional_mechanism_input(self, Wh):
        B, V, H, W, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(V, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, V, 1, 1, 1)
        all_combinations_matrix = t.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-2
        )
        return all_combinations_matrix.view(B, V, V, T, 2 * H * W)

    def spatial_batch_prepare_attentional_mechanism_input(self, Wh):
        B, V, H, W, T = Wh.shape
        Wh_repeated_in_chunks = Wh.repeat_interleave(V, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, V, 1, 1, 1)
        all_combinations_matrix = t.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
        )
        return all_combinations_matrix.view(B, V, V, H, W, 2 * T)

    def compute_attention(self, attention_type: str, Wh):
        a_input = (
            self.spatial_batch_prepare_attentional_mechanism_input(Wh)
            if attention_type == "spatial"
            else self.temporal_batch_prepare_attentional_mechanism_input(Wh)
        )
        e = t.matmul(
            a_input,
            self.a_spatial if attention_type == "spatial" else self.a_temporal,
        )
        e = self.leakyrelu(e.squeeze(-1))
        return F.softmax(e, dim=-1)

    def spatial_forward(self, N, V, T, H, W, Wh, adj_mat_norm_d12):
        attention = self.compute_attention("spatial", Wh)
        Wh_ = []
        for i in range(V):
            at = t.zeros(N, H, W, self.out_features).to(self.W.device)
            for j in range(V):
                at += Wh[:, i, :, :, :] * attention[:, i, j, :, :].unsqueeze(
                    3
                )  # .repeat(1, 1, 1, self.out_features) # was this useful??
            Wh_.append(at)
        h_prime = t.stack((Wh_))
        h_prime = (
            h_prime.permute(1, 3, 4, 2, 0).contiguous()
            # .view(N, H, W * self.out_features, V) was this useful??
        )
        # ipdb.set_trace()
        h_prime = t.matmul(h_prime, adj_mat_norm_d12).permute(0, 1, 3, 2, 4)
        # .view( # this was the bug!
        #    N, H, W, self.out_features, V
        # )
        # ipdb.set_trace()
        return F.elu(h_prime)

    def temporal_forward(self, N, V, T, H, W, Wh, adj_mat_norm_d12):
        attention = self.compute_attention("temporal", Wh)
        attention = t.diag_embed(attention)
        Wh = Wh.view(N, V, H * W, T)  # Warning, T might be undefined
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

    def forward(self, h):
        if len(h.size()) == 5:
            N, H, W, T, V = h.size()
            h = h.permute(0, 4, 1, 2, 3)
        else:
            N, V, H, W = h.size()

        if self.A.device != self.B.device:
            self.A = self.A.to(self.B.device)

        Wh = self.mapping(h)

        # Learnable Adjacency Matrix
        adj_mat = None
        adj_mat = self.B[:, :] + self.A[:, :]
        adj_mat_min = t.min(adj_mat)
        adj_mat_max = t.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)
        D = Variable(t.diag(t.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = t.sqrt(t.inverse(D))
        adj_mat_norm_d12 = t.matmul(t.matmul(D_12, adj_mat), D_12)
        if self.attention_type == "spatial":
            h_prime = self.spatial_forward(N, V, T, H, W, Wh, adj_mat_norm_d12)
        elif self.attention_type == "temporal":
            h_prime = self.temporal_forward(
                N, V, T, H, W, Wh, adj_mat_norm_d12
            )
        elif self.attention_type == "multi_stream":
            h_prime_spatial = self.spatial_forward(
                N, V, T, H, W, Wh, adj_mat_norm_d12
            )
            h_prime_temporal = self.temporal_forward(
                N, V, T, H, W, Wh, adj_mat_norm_d12
            )
            h_prime = t.mean(
                t.stack((h_prime_temporal, h_prime_spatial), dim=-1),
                dim=-1,
            )
        return h_prime
"""
