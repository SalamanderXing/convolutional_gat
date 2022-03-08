import numpy as np
import torch as t
import h5py
from ..preprocessing.utils import listdir
import matplotlib.pyplot as plt
import ipdb
from tqdm import tqdm
import json
import os
from argparse import ArgumentParser


def get_frequencies_gat_dataset(
    *, out_dir: str, data_folder: str = "/mnt/kmni_dataset/20_latest",
):
    with open(os.path.join(data_folder, "metadata.json")) as f:
        metadata = json.load(f)
    max_val = metadata["max"]
    data_dir = os.path.join(data_folder, "train")
    file_name = listdir(data_dir)
    values = {}
    for fn, fp in tqdm(file_name):
        keys, freqs = t.unique(t.load(fp), return_counts=True)
        keys = keys / max_val
        for (key, freq) in zip(keys, freqs):
            key = key.item()
            freq = freq.item()
            if key not in values:
                values[key] = 0
            values[key] += freq

    with open(os.path.join(out_dir, "stats_gat.json"), "w") as f:
        json.dump(values, f)


def get_frequencies_unet_dataset(
    *,
    out_dir: str,
    data_location: str = "/mnt/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_20.h5",
):
    dataset = h5py.File(data_location, "r")["train"]["images"]
    values = {}
    for im in tqdm(dataset):
        values_with_freq = np.asarray(np.unique(im, return_counts=True)).T
        for (key, freq) in values_with_freq:
            if key not in values:
                values[key] = 0
            values[key] += freq

    with open(os.path.join(out_dir, "stats_unet.json"), "w") as f:
        json.dump(values, f)


def plot_freqs(
    in_dir: str,
    freqs_files: tuple[str, ...] = ("stats_gat.json", "stats_unet.json"),
):
    all_freqs = []
    for file_name in freqs_files:
        with open(os.path.join(in_dir, file_name)) as f:
            all_freqs.append(
                sorted(
                    [
                        (float(key), float(val))
                        for key, val in json.load(f).items()
                    ],
                    key=lambda x: x[0],
                )
            )

    _fig, axes = plt.subplots(nrows=len(all_freqs))
    for (ax, freqs, name) in zip(
        axes, all_freqs, [fn.split(".")[0] for fn in freqs_files]
    ):
        vals, fr = list(zip(*freqs))
        max_val = round(max(vals), 4)
        max_freq = round(max(fr), 4)
        min_freq = round(min(fr), 4)
        vals = t.tensor(vals)
        fr = t.tensor(fr)
        fr = t.log(fr)
        ax.plot(vals, fr)
        ax.set_title(
            f"{name.split('_')[1].upper()} {max_val=} {max_freq=:.2e} {min_freq=}"
        )
    plt.suptitle("Values (x) vs Log Frequency (y)")
    plt.show()


def main():
    unet_file_name = "/home/wt632036/data/precipitation/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_20.h5"
    out_dir = os.path.dirname(__file__)
    get_frequencies_unet_dataset(out_dir=out_dir, data_location=unet_file_name)
    # output of this already written
    # get_frequencies_gat_dataset(out_dir=out_dir, data_folder="")
    plot_freqs(out_dir)


if __name__ == "__main__":
    main()
