import torch as t
import netCDF4
import numpy as np
import os
from argparse import ArgumentParser
import json
import ipdb
from tqdm import tqdm


def listdir(path: str):
    return [
        (subpath, os.path.join(path, subpath))
        for subpath in sorted(os.listdir(path))
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


def nested_tensor_list_to_tensor(nested):
    if len(nested) == 0:
        ipdb.set_trace()
    if type(nested[0]) in (list, tuple):
        flattened = [nested_tensor_list_to_tensor(n) for n in nested]
        return t.stack(flattened)
    else:
        return t.stack(nested)


def preprocess(
    verbose: bool = True,
    in_path: str = "~/downloads/mai_dataset",
    out_path: str = "./preprocessed",
    select_variables: list[list[str]] = [
        ["CTTH", "temperature"],
    ],
):

    log = Logger(verbose)
    conditions = ["training", "validation", "test"]
    for condition in conditions:
        log(f"Preprocessing {condition}")
        out_condition_path = os.path.join(out_path, condition)
        mkdir(out_condition_path)
        days = [d[0] for d in listdir(os.path.join(in_path, "R1", condition))]
        pdbar = tqdm(total=len(days))
        day_idx = 0
        frame_count = 4
        while day_idx < len(days):
            accumulator = []
            for rel_region_path, region_path in listdir(in_path):
                region_accumulator = [[] for _ in select_variables]
                accumulator.append(region_accumulator)
                current_days = days[day_idx : day_idx + 1]
                for day in current_days:
                    day_idx += 1
                    pdbar.update(1)
                    for variable_folder, variable_name in select_variables:
                        in_variable_path = os.path.join(
                            region_path, condition, day, variable_folder
                        )
                        # out_variable_path = os.path.join(out_day, variable_folder)
                        # kdir(out_variable_path)
                        max_count = 86
                        count = 0
                        for rel_file_path, file_path in listdir(
                            in_variable_path
                        ):
                            # ipdb.set_trace()
                            if count == max_count:
                                break
                            file_content = netCDF4.Dataset(file_path)
                            data = np.array(file_content[variable_name][:])
                            tensor_data = t.from_numpy(data)
                            region_accumulator[
                                select_variables.index(
                                    [variable_folder, variable_name]
                                )
                            ].append(tensor_data)
                            count += 1

            tensorized_accumulator = (
                nested_tensor_list_to_tensor(accumulator)
                .transpose(1, 2)
                .contiguous()
            )
            file_name = os.path.join(out_condition_path, f"{day_idx}.pt")
            t.save(tensorized_accumulator, file_name)


def preprocess_old(
    verbose: bool = True,
    in_path: str = "~/downloads/mai_dataset",
    out_path: str = "./preprocessed",
    select_variables: tuple[tuple[str, str]] = (("CTTH", "temperature"),),
):
    log = Logger(verbose)
    mkdir(out_path)
    region_paths = listdir(in_path)
    norm_max_min = [(0, 0, 0, 0) for s in select_variables]
    for rel_region_path, region_path in region_paths:  # R1, R2, ...
        out_region_path = os.path.join(out_path, rel_region_path)
        mkdir(out_region_path)
        log(f"Converting region {rel_region_path}")
        for rel_settings, setting in listdir(
            region_path
        ):  # train, test, validation
            out_settings = os.path.join(out_region_path, rel_settings)
            mkdir(out_settings)
            for rel_day, day in sorted(
                listdir(setting), key=lambda x: int(x[0])
            ):  # 202012, ....
                out_day = os.path.join(out_settings, rel_day)
                mkdir(out_day)
                variable_dimension = []
                for variable_folder, variable_name in select_variables:
                    in_variable_path = os.path.join(day, variable_folder)
                    # out_variable_path = os.path.join(out_day, variable_folder)
                    # kdir(out_variable_path)
                    for rel_file_path, file_path in listdir(in_variable_path):
                        # ipdb.set_trace()
                        file_content = netCDF4.Dataset(file_path)
                        data = np.array(file_content[variable_name][:])
                        tensor_data = t.from_numpy(data)
                        """
                        new_file_name = os.path.join(
                            out_variable_path,
                            rel_file_path.replace(".nc", ".pt"),
                        )
                        t.save(tensor_data, new_file_name)
                        """
                        variable_dimension.append(tensor_data)
                        # norm_max_min[] = (max(), min())
                    stacked_variable_dimension = t.stack(variable_dimension)
                    out_file_name = os.path.join(
                        out_day, f"{variable_name}_{variable_folder}.pt"
                    )
                    t.save(stacked_variable_dimension, out_file_name)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-path", type=str, default="/home/bluesk/downloads/mai_dataset"
    )
    parser.add_argument("--out-path", type=str, default="./preprocessed")
    parser.add_argument(
        "--select-variables", type=str, default='[["CTTH", "temperature"]]'
    )
    args = parser.parse_args()
    preprocess(
        in_path=args.in_path,
        out_path=args.out_path,
        select_variables=json.loads(args.select_variables),
    )


if __name__ == "__main__":
    main()
