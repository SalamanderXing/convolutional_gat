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


"""
def fix_sizes(tensors: list[t.Tensor]):
    sizes = t.tensor([t.tensor(tensor.shape) for tensor in tensors]).T
    min_sizes = t.min(sizes, dim=1)
    return t.stack()
"""

def fix_sizes(to_fix:list[t.Tensor]):
    min_size = min(f.shape[1] for f in to_fix)
    return [tensor[:, :min_size, :, :] for tensor in to_fix]

def nested_tensor_list_to_tensor(nested) -> t.Tensor:
    if len(nested) == 0:
        return t.tensor([])
    if type(nested[0]) in (list, tuple):
        flattened = [nested_tensor_list_to_tensor(n) for n in nested]
        flattened = [f for f in flattened if len(f.shape) > 1  and f.shape[1] > 0]
        return (t.stack(fix_sizes(flattened)) if len(flattened[0].shape) == 4 else t.stack(flattened)) if len(flattened) > 0 else t.tensor([])
    else:
        return t.stack(nested)


def preprocess(
    verbose: bool = True,
    in_path: str = "~/downloads/mai_dataset",
    out_path: str = "./preprocessed",
    select_variables: list[list[str]] = [["CTTH", "temperature"],],
):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    log = Logger(verbose)
    summary = {}
    conditions = ["training", "validation", "test"]
    for condition in conditions:
        min_val = -t.tensor(float("Inf"))
        max_val = t.tensor(float("Inf"))
        log(f"Preprocessing {condition}")
        out_condition_path = os.path.join(out_path, condition)
        mkdir(out_condition_path)
        days = [d[0] for d in listdir(os.path.join(in_path, "R1", condition))]
        pdbar = tqdm(total=len(days))
        day_idx = 0
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
            cur_min_val = (
                t.min(tensorized_accumulator)
                if t.sum(tensorized_accumulator) != 0.0
                else min_val
            )
            cur_max_val = (
                t.max(tensorized_accumulator)
                if t.sum(tensorized_accumulator) != 0
                else max_val
            )
            min_val = t.min(min_val, cur_min_val)
            max_val = t.max(max_val, cur_max_val)
            file_name = os.path.join(out_condition_path, f"{day_idx}.pt")
            t.save(tensorized_accumulator, file_name)
            summary[condition] = {"min": float(min_val), "max": float(max_val)}
        print(f'Done {condition}')
    print('Writing summary')
    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump(summary, f)
