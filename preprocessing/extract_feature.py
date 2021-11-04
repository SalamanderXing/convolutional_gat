import torch as t
import netCDF4
import numpy as np
import os
from argparse import ArgumentParser
import json
import ipdb


def listdir(path: str):
    return [
        (subpath, os.path.join(path, subpath)) for subpath in os.listdir(path)
    ]


def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


class Logger:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, message: str):
        if self.verbose:
            print(message)


def preprocess(
    verbose: str = True,
    in_path: str = "~/downloads/mai_dataset",
    out_path: str = "./preprocessed",
    select_variables: tuple[tuple[str, str]] = (("CTTH", "temperature"),),
    normalization: str = "uniform",
):
    log = Logger(verbose)
    mkdir(out_path)
    region_paths = listdir(in_path)
    norm_max_min = {}
    for rel_region_path, region_path in region_paths:  # R1, R2, ...
        out_region_path = os.path.join(out_path, rel_region_path)
        mkdir(out_region_path)
        log(f"Converting region {rel_region_path}")
        for rel_settings, setting in listdir(
            region_path
        ):  # train, test, validation
            out_settings = os.path.join(out_region_path, rel_settings)
            mkdir(out_settings)
            for rel_day, day in listdir(setting):  # 202012, ....
                out_day = os.path.join(out_settings, rel_day)
                mkdir(out_day)
                for variable_folder, variable_name in select_variables:
                    in_variable_path = os.path.join(day, variable_folder)
                    out_variable_path = os.path.join(out_day, variable_folder)
                    mkdir(out_variable_path)
                    for rel_file_path, file_path in listdir(in_variable_path):
                        # ipdb.set_trace()
                        file_content = netCDF4.Dataset(file_path)
                        data = np.array(file_content[variable_name][:])
                        tensor_data = t.from_numpy(data)
                        new_file_name = os.path.join(
                            out_variable_path,
                            rel_file_path.replace(".nc", ".pt"),
                        )
                        t.save(tensor_data, new_file_name)
                        # norm_max_min[] = (max(), min())


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-path", type=str, default="/home/bluesk/downloads/mai_dataset"
    )
    parser.add_argument("--out-path", type=str, default="./preprocessed")
    parser.add_argument(
        "--select-variables", type=str, default='[["CTTH", "temperature"]]'
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="uniform",
        choices=["uniform", "zscore"],
    )
    args = parser.parse_args()
    preprocess(
        in_path=args.in_path,
        out_path=args.out_path,
        select_variables=json.loads(args.select_variables),
        normalization=args.normalization,
    )


if __name__ == "__main__":
    main()
