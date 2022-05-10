import torch as t
from torch import nn
from .autoencoder import Encoder, Decoder
from ..baseline_model import GAT
import ipdb


class ConvGAT(nn.Module):
    def __init__(
        self,
        *,
        image_width=80,
        image_height=80,
        n_vertices=6,
        time_steps=4,
        attention_type="dummy",
        mapping_type="dummy"
    ):
        super().__init__()
        self.encoder = Encoder(4)
        self.decoder = Decoder(4)
        """
        gat_config = {
            # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
            "num_heads_per_layer": [
                4,
            ],  # other values may give even better results from the reported ones
            "num_features_per_layer": [
                256,
                256,
            ],  # 64 would also give ~0.975 uF1!
            "add_skip_connection": True,  # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
            "bias": True,  # bias doesn't matter that much
            "dropout": 0.0,  # dropout hurts the performance (best to keep it at 0)
        }
        self.gat = GAT(
            num_of_layers=len(gat_config["num_heads_per_layer"]),
            num_heads_per_layer=gat_config["num_heads_per_layer"],
            num_features_per_layer=gat_config["num_features_per_layer"],
            add_skip_connection=gat_config["add_skip_connection"],
            bias=gat_config["bias"],
            dropout=gat_config["dropout"],
            log_attention_weights=True,
        )
        """
        self.gat = GAT(input_size=400, n_vertices=6, time_steps=4)

    def forward(self, x: t.Tensor):
        x = x.squeeze().permute(0, 4, 3, 1, 2)
        n_regions = x.shape[1]
        h = t.stack(
            tuple(self.encoder(x[:, i]) for i in range(n_regions)), dim=1
        ).squeeze()
        h_prime = self.gat(h)  # [:, :, :, None, None]
        # ipdb.set_trace()
        dec_x = t.stack(
            tuple(self.decoder(h_prime[:, i]) for i in range(n_regions)), dim=1
        ).permute(0, 3, 4, 2, 1)
        return dec_x
