from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import torch as t

# from ..data_loaders.get_loaders import get_loaders
from ..GAT3D.GATMultistream import Model
from ..test import test
import json
import time
from ..utils.misc import get_number_parameters, model_classes
from ..unet_model import UnetModel
from ..lightning.utils.dataset_precip import get_oversampled_dataset
import ipdb


def get_metrics(models, models_folders, preprocessed_folder, downsample_size, device):
    results = {}
    for model_folder, model in zip(models_folders, models):
        """
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
        """
        test_loader = get_oversampled_dataset(downsample_size=downsample_size)
        n_vertices = 6 if not model_folder.endswith("v") else int(model_folder[-2])
        t0 = time.time()
        metrics = test(model, device, test_loader, nregions=n_vertices)
        t1 = time.time()
        metrics["Forward Pass Time (s)"] = (t1 - t0) / 1000
        metrics["N. Parameters"] = get_number_parameters(model)
        results[model_folder] = metrics
    return results


def plot(folders, out_path, loader, models, device, time_steps=9):
    print(f"{folders}")
    _fig, ax = plt.subplots(nrows=len(folders) + 2, ncols=time_steps)
    for model, folder in zip(models, folders):
        name = " ".join(folder.split("_")[1:])
        print(f"Parameters {name}: {get_number_parameters(model)}")

    names = [" ".join(f.split("_")[1:]) for f in folders]
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        for k in range(len(x)):
            raininess = y[k][y[k] != 0].numel() / y[k].numel()
            if raininess >= 0.1:
                preds = [
                    model(
                        x[
                            :,
                            :,
                            :,
                            :,
                            : int(name.split(" ")[-1][-2] if name.endswith("v") else 6),
                        ]
                    )[k]
                    for model, name in zip(models, names)
                ]
                # ipdb.set_trace()
                to_plot = [x[k], y[k]] + preds
                # to_plot = [t.pow(tp, 1 / loader.power) for tp in to_plot]
                for i, row in enumerate(ax):
                    for j, col in enumerate(row):
                        cur_to_plot = to_plot[i].cpu().detach().numpy()
                        col.imshow(
                            cur_to_plot[:, :, j, 1]
                            # if i < 2 or (not "unet" in names[i - 1])
                            # else cur_to_plot[j, :, :]
                        )

                row_labels = ["input", "ground truth"] + [
                    " ".join(f.split("_")[1:]) for f in folders
                ]
                for ax_, row in zip(ax[:, 0], row_labels):
                    ax_.set_ylabel(row)

                # creates a variable called col_labels that contains the labels for the columns
                col_labels = [f"frame {i + 1}" for i in range(time_steps)]
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


"""
def json_to_tex(data: dict, out_path: str):
    # ipdb.set_trace()
    result = "Model & "
    metrics_keys = list(data[list(data.keys())[0]].keys())
    result += f'{"".join(""+el.replace("_", " ") +" & " for el in metrics_keys)} \\\\ \\hline \n'
    model_keys = list(data.keys())
    for model_key in model_keys:
        result += f"{model_key.replace('_',' ').replace('local ','')} & "
        for feature_key in metrics_keys:
            feature = round(
                data[model_key][feature_key],
                0 if feature_key == "n_parameters" else 5,
            )
            result += f"{feature} & "
        result += " \\\\ \\hline \n"
    result += ""
    with open(os.path.join(out_path, "results.tex"), "w") as f:
        f.write(result)
    return result
"""


def adjust(key, val):
    result = val
    if key == "N. Parameters":
        result = round(val)
    elif key == "MSE":
        result = f"{val * (10 ** 5):.3f}e-05"
    elif key == "NMSE":
        result = f"{val:.3f}"
    else:
        result = f"{val:.4f}"
    return result


def json_to_csv(data: dict, out_path: str):
    # ipdb.set_trace()
    result = "Model,"
    metrics_keys = list(data[list(data.keys())[0]].keys())
    result += (f'{"".join(""+el.replace("_", " ") +"," for el in metrics_keys)}')[
        :-1
    ] + "\n"
    model_keys = list(data.keys())
    for model_key in model_keys:
        result += f"{model_key.replace('_',' ').replace('local ','')},"
        for i, feature_key in enumerate(metrics_keys):
            feature = adjust(feature_key, data[model_key][feature_key])
            result += f"{feature},"
        result = result[:-1]
        result += "\n"
    result += ""
    with open(os.path.join(out_path, "results.tex"), "w") as f:
        f.write(result)
    return result


def compare_models(
    base_path: str,
    folders: list[str],
    out_path="",
    downsample_size=40,
    preprocessed_folder: str = "/mnt/kmni_dataset/50_latest",
    dataset="kmni",
    plot_only=True,
    cuda=True,
    nregions=(6,),
):
    with t.no_grad():
        device = t.device("cuda" if (t.cuda.is_available() and cuda) else "cpu")
        """
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
            _, image_width, image_height, steps, _ = x.shape
            break
        """
        test_loader = get_oversampled_dataset(downsample_size=downsample_size)
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
            n_vertices = 6 if not folder.endswith("v") else int(folder[-2])
            print(f"{n_vertices=}")
            model = model_class(
                image_width=downsample_size,
                image_height=downsample_size,
                attention_type=config["MODEL_TYPE"],
                mapping_type=config["MAPPING_TYPE"],
                n_vertices=n_vertices,
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
            # print(json_to_table(results, out_path))
            # print(json_to_tex(results, out_path))
            csv = json_to_csv(results, out_path)
            print(csv)
            print(json.dumps(results, indent=4))
            with open(os.path.join(out_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(out_path, "results.csv"), "w") as f:
                f.write(csv)

        plot(folders, out_path, test_loader, models, device)


def main():
    # t.manual_seed(234)
    base_folder = "convolutional_gat/experiments"
    sizes = {"small": 20, "medium": 40, "large": 80, "xlarge": 100}
    size = "large"
    nregions = (6,)  # (3, 4, 5)
    # baseline medium
    models = [
        # "local_baseline2d",
        # "local_baseline",
        # "local_spatial_conv",
        #"local_unet",
        # "local_temporal_conv",
        # "local_multi_stream_conv",
        # "local_temporal_conv_no_group"
        # "local_conv_gat"
        # "local_temporal_conv"
        "local_convgat3d"
    ]
    proto_folders = [f"{m}_{size}" for m in models]
    if len(nregions) > 1:
        folders = []
        for n in nregions:
            for folder in proto_folders:
                folders.append(f"{folder}_{n}v")
    else:
        folders = proto_folders
    print(f"{folders=}")
    compare_models(
        base_folder,
        folders,
        out_path="convolutional_gat/compare_models/results",
        downsample_size=sizes[size],
        plot_only=False,
        cuda=False,
        nregions=nregions,
    )


if __name__ == "__main__":
    main()
