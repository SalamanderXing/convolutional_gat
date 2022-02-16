from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import torch as t
from ..data_loaders.get_loaders import get_loaders
from ..GAT3D.GATMultistream import Model
from ..train import test
import json
import time
from ..utils import get_number_parameters, model_classes
import ipdb


def get_metrics(models, models_folders, preprocessed_folder, downsample_size):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    results = {}
    for model_folder, model in zip(models_folders, models):
        train_loader, test_loader, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=100,
            preprocessed_folder=preprocessed_folder,
            device=device,
            downsample_size=downsample_size,
            dataset="kmni",
            merge_nodes=False,
            shuffle=True,
        )
        t0 = time.time()
        metrics = test(model, device, test_loader)
        t1 = time.time()
        metrics["forward_pass_time_s"] = (t1 - t0) / 1000
        metrics["n_parameters"] = get_number_parameters(model)
        results[model_folder] = metrics
    return results


def plot(out_path, loader, models):
    _fig, ax = plt.subplots(nrows=len(folders) + 1, ncols=4)
    for x, y in loader:
        for k in range(len(x)):
            raininess = y[k][y[k] != 0].numel() / y[k].numel()
            if raininess >= 0.3:
                preds = [model(x)[k] for model in models]
                to_plot = [y[k]] + preds
                to_plot = [t.pow(tp, 1 / loader.power) for tp in to_plot]
                for i, row in enumerate(ax):
                    for j, col in enumerate(row):
                        col.imshow(
                            to_plot[i].cpu().detach().numpy()[:, :, j, 1]
                        )

                row_labels = ["y"] + [
                    " ".join(f.split("_")[1:]) for f in folders
                ]
                for ax_, row in zip(ax[:, 0], row_labels):
                    ax_.set_ylabel(row)

                col_labels = ["frame1", "frame2", "frame3", "frame4"]
                for ax_, col in zip(ax[0, :], col_labels):
                    ax_.set_title(col)

                plt.savefig(os.path.join(out_path, f"multi_model_plot.png"))
                plt.close()
                return


def json_to_table(data: dict, out_path: str):
    result = "<table>"
    model_keys = list(data.keys())
    result += f'<head><th>{"".join("<td>"+el.replace("_", " ") +"</td>" for el in model_keys)}</th></head><tbody>'

    feature_keys = data[list(data.keys())[0]].keys()
    for feature_key in feature_keys:
        result += f"<tr><td>{feature_key.replace('_',' ')}</td>"
        for model_key in model_keys:
            feature = data[model_key][feature_key]
            result += f"<td>{feature:.5f}</td>"
        result += "</tr>"
    result += "</tbody></table>"
    with open(os.path.join(out_path, "results.html"), "w") as f:
        f.write(result)
    return result


def compare_models(
    base_path: str,
    folders: list[str],
    out_path="",
    downsample_size=(20, 20),
    preprocessed_folder: str = "/mnt/kmni_dataset/20_plus_preprocessed",
    dataset="kmni",
    plot_only=False,
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
        for x, y in train_loader:
            _, image_width, image_height, steps, n_vertices = x.shape
            break
        models = []
        for folder in folders:
            data_folder = os.path.join(base_path, folder)
            model_path = os.path.join(data_folder, "model.pt")
            config = {}
            exec(
                open(os.path.join(data_folder, "config.py")).read(),
                config,
            )
            model_class = model_classes[config["MODEL_TYPE"]]
            model = model_class(
                image_width=image_width,
                image_height=image_height,
                n_vertices=n_vertices,
                attention_type=config["MODEL_TYPE"],
                mapping_type=config["MAPPING_TYPE"],
            ).to(device)
            try:
                model.load_state_dict(t.load(model_path))
            except Exception:
                raise ValueError(f"error: {model_path}")
            model.eval()
            models.append(model)
        if not plot_only:
            results = get_metrics(
                models, folders, preprocessed_folder, downsample_size
            )
            print(json_to_table(results, out_path))
            print(json.dumps(results, indent=4))
            with open(os.path.join(out_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
        plot(out_path, test_loader, models)


if __name__ == "__main__":
    base_folder = "convolutional_gat/experiments"
    folders = [
        "local_temporal_conv",
        # "local_spatial_conv",
        "local_unet",
        # "local_multi_stream_conv",
    ]
    compare_models(
        base_folder,
        folders,
        out_path="convolutional_gat/compare_models/results",
        plot_only=False,
    )
