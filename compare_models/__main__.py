from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import torch as t
from ..data_loaders.get_loaders import get_loaders
from ..GAT3D.GATMultistream import Model
from ..train import test
import json
import time
from ..utils.misc import get_number_parameters, model_classes
from ..unet_model import UnetModel
import ipdb


def get_metrics(models, models_folders, preprocessed_folder, downsample_size, device):
    results = {}
    for model_folder, model in zip(models_folders, models):
        _, test_loader, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=10,
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


def plot(folders, out_path, loader, models):
    _fig, ax = plt.subplots(nrows=len(folders) + 1, ncols=4)
    for model, folder in zip(models, folders):
        name = " ".join(folder.split("_")[1:])
        print(f"Parameters {name}: {get_number_parameters(model)}")

    names = [" ".join(f.split("_")[1:]) for f in folders]
    for x, y in loader:
        for k in range(len(x)):
            raininess = y[k][y[k] != 0].numel() / y[k].numel()
            if raininess >= 0.1:
                preds = [model(x)[k] for model, name in zip(models, names)]
                # ipdb.set_trace()
                to_plot = [y[k]] + preds
                # to_plot = [t.pow(tp, 1 / loader.power) for tp in to_plot]
                for i, row in enumerate(ax):
                    for j, col in enumerate(row):
                        cur_to_plot = to_plot[i].cpu().detach().numpy()
                        col.imshow(
                            cur_to_plot[:, :, j, 1]
                            # if i < 2 or (not "unet" in names[i - 1])
                            # else cur_to_plot[j, :, :]
                        )

                row_labels = ["y"] + [" ".join(f.split("_")[1:]) for f in folders]
                for ax_, row in zip(ax[:, 0], row_labels):
                    ax_.set_ylabel(row)

                col_labels = ["frame1", "frame2", "frame3", "frame4"]
                for ax_, col in zip(ax[0, :], col_labels):
                    ax_.set_title(col)

                # plt.savefig(os.path.join(out_path, f"multi_model_plot.png"))
                plt.show()
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

def json_to_tex(data: dict, out_path: str):
    result = "metric & "
    model_keys = list(data.keys())
    result += f'{"".join(""+el.replace("_", " ") +" & " for el in model_keys)} \\\\ \\hline \n'
    feature_keys = data[list(data.keys())[0]].keys()
    for feature_key in feature_keys:
        result += f"{feature_key.replace('_',' ')} & "
        for model_key in model_keys:
            feature = data[model_key][feature_key]
            result += f"{feature:.5f} & "
        result += " \\\\ \\hline \n"
    result += ""
    with open(os.path.join(out_path, "results.tex"), "w") as f:
        f.write(result)
    return result



def compare_models(
    base_path: str,
    folders: list[str],
    out_path="",
    downsample_size=(40, 40),
    preprocessed_folder: str = "/mnt/kmni_dataset/50_latest",
    dataset="kmni",
    plot_only=True,
    cuda="cuda",
):
    with t.no_grad():
        device = t.device("cuda" if (t.cuda.is_available() and cuda) else "cpu")
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
                open(os.path.join(data_folder, "config.py")).read(), config,
            )
            config = {key: val for key, val in config.items() if key.upper() == key}
            print(json.dumps(config, indent=4, default=str))
            model_class = (
                model_classes[config["MODEL_TYPE"]]
                # if not "unet" in config["MODEL_TYPE"]
                # else UnetModel
            )
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
                models, folders, preprocessed_folder, downsample_size, device
            )
            print(json_to_table(results, out_path))
            print(json_to_tex(results, out_path))
            print(json.dumps(results, indent=4))
            with open(os.path.join(out_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
        plot(folders, out_path, test_loader, models)


def main():
    base_folder = "convolutional_gat/experiments"
    folders = [
        "local_temporal_conv_small",
        "local_baseline",
        # "local_baseline2d"
        # "local_spatial_conv",
        # "local_unet",
        # "local_multi_stream_conv",
    ]
    compare_models(
        base_folder,
        folders,
        out_path="convolutional_gat/compare_models/results",
        plot_only=False,
        cuda=True,
    )


if __name__ == "__main__":
    main()
