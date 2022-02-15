import h5py
import torch as t
from argparse import ArgumentParser

# listdir is the same as same as os.listdir but it returns a list of tuples (file_name, absolute_file_name).
# mkdir creates a folder only if it doesn't exist already
from ..utils import listdir, mkdir
import numpy as np
import json
import ipdb
import matplotlib.pyplot as plt
import os
from pathlib import Path

# used only to display
def draw_rectangle(img, x0, y0, width, height, border=3):  # not currently used
    original = t.clone(img[x0 : x0 + width, y0 : y0 + height])
    img[x0 : x0 + width, y0 : y0 + height] = 200
    inner = original[border:-border, border:-border]
    img[
        x0 + border : x0 + width - border, y0 + border : y0 + height - border
    ] = inner


def get_z_score_normalizing_constants(preprecessed_folder: str):
    acc = t.cat(
        tuple(
            t.load(fpath)
            for fname, fpath in listdir(
                os.path.join(preprecessed_folder, "train")
            )
        )
    ).float()
    result = {
        "mean": t.mean(acc, dim=0),
        "var": t.var(acc, dim=0),
    }
    t.save(result, os.path.join(preprecessed_folder, "metadata.pt"))


def preprocess(
    in_dir: str,
    out_dir: str,
    from_year: int = 2016,
    rain_threshold: float = 0.2,
):
    out_dir = Path(out_dir) / "train"

    # Careful here (imagine someone putting in an "important" folder such as "system32")
    if os.path.exists(out_dir):
        os.removedirs(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    years = listdir(in_dir)
    acc = []  # we will use it to store the continous frame of data (video)
    coordinates = (  # coordinates of areas of interest within the larger image
        (201, 38),
        (201 - 80, 81),
        (201 - 80 + 4, 81 + 92),
        (214, 140),
        (29, 190),
        (29 + 10, 186 - 85),
    )
    file_index = 0
    patience = True
    max_val = 0
    min_val = 1000
    if from_year != -1:
        index = [y[0] for y in years].index(str(from_year))
        years = years[index:]
    for year, abs_path in years:
        print(f"Year: {year}")
        months = listdir(abs_path)
        for month, month_path in months:
            print(f"\t{month}")
            days = [
                (fname, fpath)
                for fname, fpath in listdir(month_path)
                if fname.endswith(".h5")
            ]
            for file, file_path in days:
                raw_content = t.from_numpy(
                    h5py.File(file_path)["image1"]["image_data"][...].astype(
                        np.int64
                    )
                    # .astype(  # read file, we keep it a uint8 to save memory
                    #    np.uint8
                    # )
                )
                max_val = max(t.max(raw_content).item(), max_val)
                min_val = min(t.min(raw_content).item(), min_val)
                raw_content = raw_content[
                    243:590, 234:512
                ]  # subsample the image

                # List comprehension is faster most of the time
                content_accumulator = [
                    raw_content[x : x + 80, y : y + 80] for x, y in coordinates
                ]
                content = t.stack(
                    content_accumulator
                )  # merge them into one tensor
                content[content == 65535] = 0  # set NaNs to zero
                raininess = (
                    1 - t.sum(content == 0) / content.numel()
                )  # compute raininess of single image
                if raininess >= rain_threshold:
                    acc.append(content)
                    patience = True
                elif (
                    patience
                ):  # patience allows for one frame to not be enough rainy, this increases considerably the size of the dataset
                    acc.append(content)
                    patience = False
                elif (
                    len(acc) >= 8
                ):  # select the continuous 'video' only if its length is 8, this is the minimum length we can make use of
                    tensorized_acc = t.stack(acc)
                    file_name = os.path.join(
                        out_dir, f'{str(file_index).rjust(10, "0")}.pt'
                    )
                    acc = []
                    t.save(tensorized_acc, file_name)
                    file_index += 1
                else:  # if the size is too small, discard the data
                    acc = []
            if len(acc) > 8:
                tensorized_acc = t.stack(acc)
                file_name = os.path.join(
                    out_dir, f'{str(file_index).rjust(10, "0")}.pt'
                )
                acc = []
                t.save(tensorized_acc, file_name)
                file_index += 1
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"max": max_val, "min": min_val}, f)


def test_split(out_dir: str, ratio=0.2):
    train_out_dir = os.path.join(out_dir, "train")
    test_out_dir = os.path.join(out_dir, "test")
    mkdir(test_out_dir)
    files = listdir(train_out_dir)
    randperm = t.randperm(len(files))
    test_indices = randperm[: int(len(files) * ratio)]
    for i in test_indices:
        file_name, file_path = files[i]
        os.system(f"mv {file_path} {os.path.join(test_out_dir,file_name)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "action", choices=("preprocess", "test-split", "minimize", "z-score")
    )
    parser.add_argument("-i", "--in-dir", type=str)
    parser.add_argument("-o", "--out-dir", type=str)
    parser.add_argument("-r", "--rain-threshold", type=float, default=0.5)
    parser.add_argument("-y", "--from-year", type=int, default=2016)
    args = parser.parse_args()
    assert args.rain_threshold <= 1, "--rain-threshold must be <= 1"
    print(json.dumps(args.__dict__, indent=4))
    if args.action == "preprocess":
        preprocess(
            args.in_dir, args.out_dir, args.from_year, args.rain_threshold
        )
        test_split(args.out_dir)
    elif args.action == "test-split":
        test_split(args.out_dir)
    elif args.action == "z-score":
        get_z_score_normalizing_constants(args.out_dir)
