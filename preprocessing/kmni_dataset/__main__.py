import h5py
import torch as t
from argparse import ArgumentParser
from ..utils import listdir, mkdir
import numpy as np
import json
import ipdb
import matplotlib.pyplot as plt
import os


def draw_rectangle(img, x0, y0, width, height, border=3):
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
    in_dir: str, out_dir: str, from_year: int = 2016, rain_threshold=0.2
):
    mkdir(out_dir)
    out_dir = os.path.join(out_dir, "train")
    mkdir(out_dir)
    os.system(f"rm {out_dir if out_dir.endswith('/') else out_dir + '/'}*")
    years = listdir(in_dir)
    acc = []
    rain_credit = 0
    file_index = 0
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
                        np.uint8
                    )
                )
                raw_content = raw_content[243:590, 234:512]
                coordinates = (
                    (201, 38),
                    (201 - 80, 81),
                    (201 - 80 + 4, 81 + 92),
                    (214, 140),
                    (29, 190),
                    (29 + 10, 186 - 85),
                )
                content_accumulator = []
                for x, y in coordinates:
                    # draw_rectangle(content, x, y, 80, 80)
                    content_accumulator.append(
                        raw_content[x : x + 80, y : y + 80]
                    )
                content = t.stack(content_accumulator) / 120
                content[content == 255] = 0
                raininess = 1 - t.sum(content == 0) / t.prod(
                    t.tensor(content.shape)
                )
                ipdb.set_trace()
                if raininess >= rain_threshold:
                    acc.append(content)
                elif len(acc) >= 8:
                    tensorized_acc = t.stack(acc)
                    file_name = os.path.join(
                        out_dir, f'{str(file_index).rjust(10, "0")}.pt'
                    )
                    """
                    print(
                        f"Writing file: {file_name}, {tensorized_acc.shape=}"
                    )
                    """
                    acc = []
                    t.save(tensorized_acc, file_name)
                    file_index += 1
                else:
                    acc = []
                """
                if raininess >= 0.5:
                    acc.append(content)
                    rain_credit += 1
                elif rain_credit > 0:
                    acc.append(content)
                    rain_credit -= 1
                elif len(acc) > 0:
                    tensorized_acc = t.stack(acc)
                    file_name = f'{str(file_index).rjust(10, "0")}.pt'
                    print(f"Writing file: {file_name}, {len(acc)=}")
                    acc = []
                    t.save(tensorized_acc, file_name)
                    file_index += 1
                """


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


def get_mini_dataset(in_dir: str, out_dir: str):
    mkdir(out_dir)
    for cond in ("test", "train"):
        cond_in = os.path.join(in_dir, cond)
        cond_out = os.path.join(out_dir, cond)
        files = listdir(cond_in)
        for file_name, file_path in files:
            data = t.load(file_path)
            shrunk_data = data[:, :, 25:-25, 25:-25]
            shrunk_file_name = os.path.join(cond_out, file_name)
            t.save(shrunk_data, shrunk_file_name)


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
    elif args.action == "minimize":
        get_mini_dataset(args.in_dir, args.out_dir)
    elif args.action == "z-score":
        get_z_score_normalizing_constants(args.out_dir)
