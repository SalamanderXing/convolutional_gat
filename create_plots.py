import torch as t
import torch.nn as nn
from torchinfo import summary
import os
from tqdm import tqdm
import json
import ipdb
from argparse import ArgumentParser
from .data_loaders.get_loaders import get_loaders
from .model import SpatialModel, TemporalModel
import matplotlib.pyplot as plt
from .unet_model import UnetModel
from convolutional_gat.baseline_model import BaselineModel, BaselineModel2D


def create_comparison_plot(
    models,
    epoch=1,
    path="",
    downsample_size=(256, 256),
    preprocessed_folder: str = "",
    dataset="kmni",
    max_preds=1,
):
    plt.clf()
    with t.no_grad():
        N_COLS = 4  # frames
        N_ROWS = 2 + len(models)  # x, y, preds
        # plt.title(f"Epoch {epoch}")
        _fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS)
        # for model_file in enumerate(models):
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        # merge_nodes = issubclass(UnetModel, type(model))
        merge_nodes = False
        loader, _, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=2,
            preprocessed_folder="convolutional_gat/preprocessed",
            device=device,
            downsample_size=downsample_size,
            dataset=dataset,
            merge_nodes=merge_nodes,
        )
        for x, y in loader:
            for k in range(len(x)):
                raininess = t.sum(x[k] != 0) / t.prod(t.tensor(x[k].shape))
                if raininess >= 0.5:
                    # preds = model(x)
                    # to_plot = [x[k], y[k], preds[k]]
                    to_plot = [x[k], y[k]]
                    row_labels = ["x", "y"]
                    for model_obj, model_file in models:
                        model = model_obj
                        model.load_state_dict(
                            t.load(
                                os.getcwd()
                                + "/convolutional_gat/experiments/"
                                + model_file
                                + "/model.pt"
                            )
                        )
                        # model = t.jit.load()
                        model.to(device)
                        model.eval()
                        preds = model(x)
                        to_plot.append(preds[k])
                        row_labels.append(model_file)

                    for i, row in enumerate(ax):
                        for j, col in enumerate(row):
                            # ipdb.set_trace()
                            col.imshow(
                                to_plot[i].cpu().detach().numpy()[:, :, j, 1]
                                if not merge_nodes
                                else to_plot[i]
                                .cpu()
                                .detach()
                                .numpy()[j, : downsample_size[0], : downsample_size[1],]
                            )

                    for ax_, row in zip(ax[:, 0], row_labels):
                        ax_.set_ylabel(row)

                    col_labels = ["frame1", "frame2", "frame3", "frame4"]
                    for ax_, col in zip(ax[0, :], col_labels):
                        ax_.set_title(col)

                    plt.savefig(
                        f"{os.getcwd()}/convolutional_gat/models_comparison/pred_{k}.png"
                    )
                    plt.close()
                    # model.train()

                    if (k + 1) == max_preds:
                        return


if __name__ == "__main__":
    create_comparison_plot(
        models=[
            (
                BaselineModel(
                    image_width=20,
                    image_height=20,
                    n_vertices=6,
                    mapping_type="linear",
                ),
                "final_gat1d",
            ),
            (
                BaselineModel2D(
                    image_width=20,
                    image_height=20,
                    n_vertices=6,
                    mapping_type="linear",
                ),
                "final_gat2d",
            ),
        ],
        epoch=1,
        path="",
        downsample_size=(20, 20),
        max_preds=1,
    )
