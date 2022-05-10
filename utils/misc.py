import h5py
import numpy as np
import torch as t
import ipdb
import enum
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from ..baseline_model import BaselineModel, BaselineModel2D
from ..GAT3D.conv_gat import ConvGAT
from ..GAT3D.GATMultistream import Model as GatModel
from ..GAT3D.conv_gat3d import ConvGAT3D

# from ..data_loaders.get_loaders import get_loaders
import os


# from ..unet_model import UnetModel
from ..lightning.models.unet_precip_regression_lightning import UNetDS_Attention


class UnetModel(UNetDS_Attention):
    def __init__(
        self,
        image_width,
        image_height,
        n_vertices,
        attention_type,
        mapping_type,
        n_heads_per_layer=(1,),
    ):
        hparams = ArgumentParser().parse_args()
        hparams.__dict__.update(
            {
                "model": "UNetDS_Attention",
                "n_channels": 9,
                "n_classes": 9,
                "kernels_per_layer": 2,
                "bilinear": True,
                "reduction_ratio": 16,
                "lr_patience": 4,
                "num_input_images": 4,
                "num_output_images": 4,
                "valid_size": 0.1,
                "use_oversampled_dataset": True,
                "logger": True,
                "checkpoint_callback": True,
                "early_stop_callback": False,
                "default_root_dir": None,
                "gradient_clip_val": 0,
                "process_position": 0,
                "num_nodes": 1,
                "num_processes": 1,
                "gpus": 1,
                "auto_select_gpus": False,
                "num_tpu_cores": None,
                "log_gpu_memory": None,
                "progress_bar_refresh_rate": 1,
                "overfit_pct": 0.0,
                "track_grad_norm": -1,
                "check_val_every_n_epoch": 1,
                "fast_dev_run": None,
                "accumulate_grad_batches": 1,
                "max_epochs": 1000,
                "min_epochs": 1,
                "max_steps": None,
                "min_steps": None,
                "train_percent_check": 1.0,
                "val_percent_check": 1.0,
                "test_percent_check": 1.0,
                "val_check_interval": 1.0,
                "log_save_interval": 100,
                "row_log_interval": 10,
                "distributed_backend": None,
                "precision": 32,
                "print_nan_grads": False,
                "weights_summary": "full",
                "weights_save_path": None,
                "num_sanity_val_steps": 2,
                "truncated_bptt_steps": None,
                "resume_from_checkpoint": None,
                "profiler": None,
                "benchmark": False,
                "deterministic": False,
                "reload_dataloaders_every_epoch": False,
                "auto_lr_find": False,
                "replace_sampler_ddp": True,
                "progress_bar_callback": True,
                "terminate_on_nan": False,
                "auto_scale_batch_size": False,
                "amp_level": "O1",
                "dataset_folder": "convolutional_gat/data/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5",
                "experiment_save_path": "local_temporal_conv",
                "batch_size": 6,
                "learning_rate": 0.001,
                "epochs": 200,
                "es_patience": 30,
                "save_path": "/home/bluesk/Documents/convolutional_gat/lightning/../experiments/local_unet",
            }
        )
        super().__init__(hparams=hparams)


model_classes = {
    "unet": UnetModel,
    "temporal": GatModel,
    "spatial": GatModel,
    "multi_stream": GatModel,
    "baseline": BaselineModel,
    "baseline_2d": BaselineModel2D,
    "convgat": ConvGAT,
    "convgat3d": ConvGAT3D,
}


def get_number_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def term_display(y, y_hat):

    plt.clf()

    if len(y.shape) == 4:
        im1 = y[0, 0, :20, :20].detach().cpu()
        im2 = y_hat[0, 0, :20, :20].detach().cpu()
    else:
        im1 = y[0, 0, 0].detach().cpu()
        im2 = y_hat[0, 0, 0].detach().cpu()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()
    _fig, ax = plt.subplots(nrows=1, ncols=2)
    ims = [im1, im2]

    for i, col in enumerate(ax):
        col.imshow(ims[i])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("/tmp/im1.png")
    print(climage.convert("/tmp/im1.png", is_unicode=True,))


def visualize_predictions(
    model,
    epoch=1,
    path="",
    downsample_size=(256, 256),
    preprocessed_folder: str = "",
    dataset="kmni",
):
    plt.clf()
    with t.no_grad():
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        train_loader, test_loader, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=2,
            preprocessed_folder=preprocessed_folder,
            device=device,
            downsample_size=downsample_size,
            dataset=dataset,
            merge_nodes=False,
            shuffle=True,
        )
        model.eval()
        N_COLS = 4  # frames
        N_ROWS = 3  # x, y, preds
        plt.title(f"Epoch {epoch}")
        _fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS)
        for x, y in test_loader:
            for k in range(len(x)):
                raininess = t.sum(x[k] > 0.0) / t.prod(t.tensor(x[k].shape))
                if raininess >= 0.2:
                    preds = model(x)
                    to_plot = [
                        t.pow(val, 1 / test_loader.power)
                        for val in [x[k], y[k], preds[k]]
                    ]
                    for i, row in enumerate(ax):
                        for j, col in enumerate(row):
                            col.imshow(to_plot[i].cpu().detach().numpy()[:, :, j, 0])

                    row_labels = ["x", "y", "preds"]
                    for ax_, row in zip(ax[:, 0], row_labels):
                        ax_.set_ylabel(row)

                    col_labels = ["frame1", "frame2", "frame3", "frame4"]
                    for ax_, col in zip(ax[0, :], col_labels):
                        ax_.set_title(col)

                    save_path = os.path.join(path, f"pred_{epoch}.png")
                    plt.savefig(save_path)
                    plt.close()
                    model.train()
                    # term_display(y, preds)
                    return
    print("Raininess threshold too strict, hasn't found anything")


def plot_history(
    history: dict, title: str = "Training History", save=False, filename="history",
):
    plt.clf()
    plt.plot(
        history["train_loss"], label="Train loss",
    )
    plt.plot(
        history["val_loss"], label="Val loss",
    )
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def update_history(history: dict, data: dict):
    for key, val in data.items():
        if key not in history:
            history[key] = []
        history[key].append(val)
