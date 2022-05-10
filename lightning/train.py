import pytorch_lightning as pl
import ipdb


from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateLogger,
    EarlyStopping,
)
from .models.gat_lightning import (
    GAT3DLightning,
    GAT2DLightning,
    GAT1DLightning,
    ConvGAT,
    ConvGAT3D
)
from pytorch_lightning import loggers
import argparse
import json
from .models import unet_precip_regression_lightning as unet_regr

# import torchsummary
import os


def get_batch_size(hparams):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    elif hparams.model == "CGAT":
        net = unet_regr.CGAT(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    trainer = pl.Trainer(gpus=hparams.gpus)
    new_batch_size = trainer.scale_batch_size(net, mode="binsearch", init_val=8)
    print("New biggest batch_size: ", new_batch_size)
    return new_batch_size


def train_regression(hparams):
    if hparams.model == "UNetDS_Attention":
        net = unet_regr.UNetDS_Attention(hparams=hparams)
    elif hparams.model == "UNet_Attention":
        net = unet_regr.UNet_Attention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "CGAT":
        net = unet_regr.CGAT(hparams=hparams)
    elif hparams.model == "GAT3D":
        net = GAT3DLightning(hparams)
    elif hparams.model == "GAT2D":
        net = GAT2DLightning(hparams)
    elif hparams.model == "GAT1D":
        net = GAT1DLightning(hparams)
    elif hparams.model == "ConvGAT":
        net = ConvGAT(hparams)
    elif hparams.model == "ConvGAT3D":
        net = ConvGAT3D(hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")
    """
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd()
        + "/"
        + hparams.save_path
        + "/"
        + net.__class__.__name__
        + "/{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
        prefix=net.__class__.__name__ + "_rain_threshhold_50_",
    )
    """
    lr_logger = LearningRateLogger()
    tb_logger = loggers.TensorBoardLogger(
        save_dir=hparams.save_path, name=net.__class__.__name__
    )

    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,  # is effectively half (due to a bug in pytorch-lightning)
    )
    trainer = pl.Trainer(
        fast_dev_run=hparams.fast_dev_run,
        gpus=hparams.gpus,
        weights_summary=None,
        max_epochs=hparams.epochs,
        default_save_path=hparams.save_path,
        # checkpoint_callback=checkpoint_callback,
        early_stop_callback=earlystopping_callback,
        logger=tb_logger,
        callbacks=[lr_logger],
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        val_check_interval=hparams.val_check_interval,
        overfit_pct=hparams.overfit_pct,
    )
    trainer.fit(net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--dataset_folder",
        default="train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5",
        type=str,
    )
    parser.add_argument(
        "--experiment-save-path", type=str, default="local_temporal_conv"
    )
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)

    args = parser.parse_args()

    args.fast_dev_run = False  # True
    models = {
        "ConvGAT": "local_convgat",
        "ConvGAT3D": "local_convgat3d",
        "UNetDS_Attention": "local_unet",
        "GAT3D": "local_temporal_conv",
        "GAT2D": "local_baseline2d",
        "GAT1D": "local_baseline",
    }
    sizes = {"small": 20, "medium": 40, "large": 80, "xlarge": 100}
    args.num_input_images = 9
    args.num_output_images = 9
    args.size = "large"
    args.subsample_size = sizes[args.size]
    args.n_channels = 9
    args.gpus = 1
    model_index = -1
    args.model = list(models.keys())[1]  #
    args.nregions = 6
    args.batch_size = 6  # if args.model == "GAT3D" else 2
    if args.model == list(models.keys())[1]:
        args.batch_size = 1
    args.lr_patience = 4
    args.es_patience = 30
    # args.val_check_interval = 0.25
    # args.overfit_pct = 0.1
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.save_path = os.path.join(
        os.path.dirname(__file__),
        f"../experiments/{models[args.model]}_{args.size}{'_' + str(args.nregions) + 'v' if args.nregions < 6 else ''}",
    )
    assert os.path.exists(args.save_path), f"Save path {args.save_path} does not exist!"
    args.experiment_save_path = args.save_path
    # args.dataset_folder = "data/precipitation/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"
    # args.resume_from_checkpoint = f"lightning/precip_regression/{args.model}/UNetDS_Attention.ckpt"
    print(json.dumps(args.__dict__, indent=4))
    train_regression(args)
