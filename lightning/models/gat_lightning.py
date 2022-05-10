import torch as t
from ...GAT3D.GATMultistream import Model as GAT3D
from ...baseline_model import BaselineModel as GAT1D, BaselineModel2D as GAT2D
from .regression_lightning import Precip_regression_base
from ...GAT3D.conv_gat import ConvGAT as PlainConvGAT
from ...GAT3D.conv_gat3d import ConvGAT3D as RawConvGAT3D


class ConvGAT3D(Precip_regression_base):
    def __init__(self, params):
        super().__init__(params)
        # self.model = GATMultiHead3D(4, 4, 0.2, 1, n_vertices=6, image_width=80, image_height=80,)
        self.model = RawConvGAT3D(params.subsample_size, params.subsample_size)

    def forward(self, x):
        return self.model(x).squeeze()

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output["loss"]
        loss_mean /= len(outputs)

        if loss_mean < self.min_loss:
            self.min_loss = loss_mean
            t.save(self.model.cpu().state_dict(), self.save_path)
            self.model.cuda()
        return {
            "log": {"train_loss": loss_mean},
            "progress_bar": {"train_loss": loss_mean},
        }


class ConvGAT(Precip_regression_base):
    def __init__(self, params):
        super().__init__(params)
        # self.model = GATMultiHead3D(4, 4, 0.2, 1, n_vertices=6, image_width=80, image_height=80,)
        self.model = PlainConvGAT()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output["loss"]
        loss_mean /= len(outputs)

        if loss_mean < self.min_loss:
            self.min_loss = loss_mean
            t.save(self.model.cpu().state_dict(), self.save_path)
            self.model.cuda()
        return {
            "log": {"train_loss": loss_mean},
            "progress_bar": {"train_loss": loss_mean},
        }


class GAT3DLightning(Precip_regression_base):
    def __init__(self, params):
        super().__init__(params)
        # self.model = GATMultiHead3D(4, 4, 0.2, 1, n_vertices=6, image_width=80, image_height=80,)
        self.model = GAT3D(
            image_width=params.subsample_size,
            image_height=params.subsample_size,
            n_vertices=params.nregions,
            attention_type="temporal",
            mapping_type="conv",  # "smaat_unet",
        )

    def forward(self, x):
        return self.model(x).squeeze()

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output["loss"]
        loss_mean /= len(outputs)

        if loss_mean < self.min_loss:
            self.min_loss = loss_mean
            t.save(self.model.cpu().state_dict(), self.save_path)
            self.model.cuda()
        return {
            "log": {"train_loss": loss_mean},
            "progress_bar": {"train_loss": loss_mean},
        }


class GAT2DLightning(Precip_regression_base):
    def __init__(self, params):
        super().__init__(params)
        # self.model = GATMultiHead3D(4, 4, 0.2, 1, n_vertices=6, image_width=80, image_height=80,)
        self.model = GAT2D(
            image_width=params.subsample_size,
            image_height=params.subsample_size,
            n_vertices=6,
            attention_type="temporal",
            mapping_type="conv",
        )

    def forward(self, x):
        return self.model(x).squeeze()

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output["loss"]
        loss_mean /= len(outputs)

        if loss_mean < self.min_loss:
            self.min_loss = loss_mean
            t.save(self.model.cpu().state_dict(), self.save_path)
            self.model.cuda()
        return {
            "log": {"train_loss": loss_mean},
            "progress_bar": {"train_loss": loss_mean},
        }


class GAT1DLightning(Precip_regression_base):
    def __init__(self, params):
        super().__init__(params)
        # self.model = GATMultiHead3D(4, 4, 0.2, 1, n_vertices=6, image_width=80, image_height=80,)
        self.model = GAT1D(
            image_width=params.subsample_size,
            image_height=params.subsample_size,
            n_vertices=6,
            attention_type="temporal",
            mapping_type="conv",
        )

    def forward(self, x):
        return self.model(x).squeeze()

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output["loss"]
        loss_mean /= len(outputs)

        if loss_mean < self.min_loss:
            self.min_loss = loss_mean
            t.save(self.model.cpu().state_dict(), self.save_path)
            self.model.cuda()
        return {
            "log": {"train_loss": loss_mean},
            "progress_bar": {"train_loss": loss_mean},
        }
