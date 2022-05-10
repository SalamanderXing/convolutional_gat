import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .resnet_autoencoder import ResNet18Dec, ResNet18Enc
from .GATMultistream import Model as GAT3D
import ipdb

class ConvGAT3D(nn.Module):
    def __init__(self, image_height: int, image_width: int, *, attention_type="", mapping_type="", n_vertices=6, lag=9):
        super().__init__()
        self.encoder = ResNet18Enc(nc=lag)
        self.decoder = ResNet18Dec(nc=lag)
        self.gat3d = GAT3D(
            image_height=16,
            image_width=16,
            n_vertices=n_vertices,
            time_steps=lag,
            mapping_type='conv',
            attention_type='temporal'
        )

    def forward(self, x):
        x_p = x.permute(0, 4, 3, 1, 2)
        enc_x = t.stack(tuple(self.encoder(x_p[:,i]) for i in range(x_p.shape[1])), dim=1)
        gat_x = self.gat3d(enc_x.permute(0, 3, 4, 2, 1))
        gat_x_p = gat_x.permute(0, 4, 3, 1, 2)
        dec_x = t.stack(tuple(self.decoder(gat_x_p[:,i]) for i in range(x_p.shape[1])), dim=1)
        x_p = dec_x.permute(0, 3, 4, 2, 1)        
        return x_p
