import torch as t
import netCDF4
import numpy as np
import os
from argparse import ArgumentParser


def listdir(path: str):
    return [os.path.join(path, subpath) for subpath in os.listdir(path)]


def preprocess(
    in_path: str = "~/downloads/mai_dataset", out_path: str = "./preprocessed"
):
    dataset_path = region_paths = listdir(in_path)
    for region_path in region_paths:
        for setting in listdir(region_path):  # train, test, validation
            for days in listdir(setting):
                datasets = []
                for var_dir in var_dirs:
                    cur_dir = os.path.join(root_dir, var_dir)
                    fpath = os.path.join(cur_dir, os.listdir(cur_dir)[0])
                    dataset = netCDF4.Dataset(fpath)
                    datasets.append(dataset)
                    variables = dataset.variables.keys()
                    print(variables)


def main():
    parser = ArgumentParser()
    parser.add_argument("in_path", type=str, default="~/downloads/mai_dataset")
    parser.add_argument("out_path", type=str, default="./preprocessed")
    args = parser.parse_args()
    preprocess(in_path=args.in_path, out_path=args.out_path)


if __name__ == "__main__":
    main()
