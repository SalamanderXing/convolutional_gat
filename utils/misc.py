import h5py
import numpy as np
import torch as t
import ipdb
import enum
import matplotlib.pyplot as plt
import climage
from ..unet_model import UnetModel
from ..GAT3D.GATMultistream import Model as GatModel
from ..data_loaders.get_loaders import get_loaders
import os

model_classes = {
    "unet": UnetModel,
    "temporal": GatModel,
    "spatial": GatModel,
    "multi_stream": GatModel,
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
                            col.imshow(
                                to_plot[i].cpu().detach().numpy()[:, :, j, 0]
                            )

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
    history: dict[str, list[float]],
    title: str = "Training History",
    save=False,
    filename="history",
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


def update_history(history: dict[str, list[float]], data: dict[str, float]):
    for key, val in data.items():
        if key not in history:
            history[key] = []
        history[key].append(val)
