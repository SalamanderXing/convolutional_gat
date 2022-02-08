import json
import os

import ipdb
import netCDF4
import numpy as np
import torch as t
from tqdm import tqdm

from ..utils import listdir, mkdir

# TODO: normalized data
# TODO: get precipitation instead of temperature
# TODO: implement blacklisting


class Logger:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, message: str):
        if self.verbose:
            print(message)


def fix_sizes(to_fix: list[t.Tensor]) -> list[t.Tensor]:
    min_size = min(f.shape[1] for f in to_fix)
    return [tensor[:, :min_size, :, :] for tensor in to_fix]


def get_time_range() -> tuple[int, ...]:
    acc: list[tuple[int, int]] = [(0, 0)]
    for _ in range(1, 96):
        tmp = acc[-1][1] + 15
        if tmp == 60:
            acc.append((acc[-1][0] + 1, 0))
        else:
            acc.append((acc[-1][0], tmp))
    result = tuple(int(f"{i}{j if j != 0 else '00'}") for i, j in acc)
    return result


def get_time_from_file_name(file_name: str) -> int:
    return int(file_name.split("_")[-1].split("T")[-1].strip("Z.nc")[:-2])


def get_missing_mask(directory: str) -> tuple[str, ...]:
    _, files = tuple(zip(*listdir(directory)))
    files = sorted(files, key=get_time_from_file_name)
    files_time_steps = tuple(get_time_from_file_name(fn) for fn in files)
    all_time_steps = get_time_range()
    fixed_files: list[str] = []
    i = 0
    for time_step in all_time_steps:
        if i < len(files_time_steps) and time_step == files_time_steps[i]:
            fixed_files.append(files[i])
            i += 1
        else:
            fixed_files.append("")
    return tuple(fixed_files)


def get_continuous_splits(directory: str) -> list[list[str]]:
    files = get_missing_mask(directory)
    acc = [[]]
    for f in files:
        if f != "":
            acc[-1].append(f)
        else:
            acc.append([])
    return acc


def nested_tensor_list_to_tensor(nested) -> t.Tensor:
    if len(nested) == 0:
        return t.tensor([])
    if type(nested[0]) in (list, tuple):
        flattened = [nested_tensor_list_to_tensor(n) for n in nested]
        flattened = [f for f in flattened if len(f.shape) > 1 and f.shape[1] > 0]
        return (
            (
                t.stack(fix_sizes(flattened))
                if len(flattened[0].shape) == 4
                else t.stack(flattened)
            )
            if len(flattened) > 0
            else t.tensor([])
        )
    else:
        return t.stack(nested)


def merge(files, new_files):
    if len(files) > 0:
        files[-1] += new_files[0]
    files += new_files[1:]


def merge_days(accumulator: dict,) -> dict[str, dict[tuple[str, str], list[list[str]]]]:
    fixed_acc = {}
    for region, var_acc in accumulator.items():
        fixed_acc[region] = {}
        for var, days_acc in var_acc.items():
            days = tuple(sorted((d for d in days_acc.keys()), key=int))
            new_days_acc = []
            merge(new_days_acc, days_acc[days[0]])
            for i, day in enumerate(days[1:]):
                if int(day) != (int(days[i]) + 1):
                    new_days_acc.append([])
                merge(new_days_acc, days_acc[day])
            fixed_acc[region][var] = [
                merged_files for merged_files in new_days_acc if len(merged_files) > 0
            ]
    return fixed_acc


def split_continuous_blocks_at_root(
    accumulator: dict[str, dict[tuple[str, str], list[list[str]]]]
) -> list[dict[str, dict[tuple[str, str], list[str]]]]:
    new_accumulator = []
    a_region = tuple(accumulator.keys())[0]
    a_variable = tuple(accumulator[a_region].keys())[0]
    for i in range(len(accumulator[a_region][a_variable])):
        current_root = {}
        new_accumulator.append(current_root)
        for region, vars_acc in accumulator.items():
            region_accumulator = {}
            current_root[region] = region_accumulator
            for var, blocks in vars_acc.items():
                region_accumulator[var] = accumulator[region][var][i]
    return new_accumulator


def block_to_tensor(block: dict[str, dict[tuple[str, str], list[str]]]) -> t.Tensor:
    accumulator = []
    for _, var_acc in block.items():
        region_accumulator = []
        for var, block_files in var_acc.items():
            files_accumulator = []
            for file in block_files:
                fc = netCDF4.Dataset(file)[var[1]]
                masked_array = fc[...]
                valid_range = fc.valid_range
                """
                print(
                    f"{np.sum(masked_array.mask) / len(masked_array.mask.flatten())=}"
                )
                """
                array = masked_array.filled(
                    (np.max(valid_range) - np.min(valid_range)) / 2
                )
                scale_factor = fc.scale_factor if "scale_factor" in fc.__dict__ else 1
                add_offset = fc.add_offset if "add_offset" in fc.__dict__ else 0
                normalized_array = (
                    (array / (np.max(valid_range) * scale_factor)) - add_offset
                ).astype(np.float32)
                # minv = np.min(fc)
                # maxv = np.max(fc)
                # if minv != 0 or maxv != 0:
                """
                print(f"{maxv=}")
                print(f"{minv=}")
                print(f"{np.max(normalized_array)=}")
                print(f"{np.min(normalized_array)=}")
                """
                data = t.from_numpy(normalized_array)
                files_accumulator.append(data)
            tensorized_file_accumulator = t.stack(files_accumulator)
            region_accumulator.append(tensorized_file_accumulator)
        tensorized_region_accumulator = t.stack(region_accumulator)
        accumulator.append(tensorized_region_accumulator)
    tensorized_accumulator = t.stack(accumulator).permute(2, 0, 1, 3, 4)
    return tensorized_accumulator


def preprocess(
    verbose: bool = True,
    lag: int = 4,
    seq_size: int = 4,
    in_path: str = "~/downloads/mai_dataset",
    out_path: str = "./preprocessed",
    select_variables: tuple[tuple[str, str], ...] = (("CRR", "crr"),),
):
    if os.path.exists(out_path):
        os.system(f"rm -rf {out_path}")
    os.mkdir(out_path)
    print(f"{in_path=}")
    print(f"{out_path=}")
    log = Logger(verbose)
    n_regions = len(os.listdir(in_path))
    print(f"{n_regions=}")
    metadata = {"n_regions": n_regions}
    conditions = ["training", "validation"]
    for condition in conditions:
        metadata[condition] = {"length": 0}
        log(f"Preprocessing {condition}")
        out_condition_path = os.path.join(out_path, condition)
        mkdir(out_condition_path)
        days = sorted(
            (d[0] for d in listdir(os.path.join(in_path, "R1", condition))), key=int,
        )
        accumulator = {}
        for day in tqdm(days):
            for rel_region_path, region_path in listdir(in_path):
                if rel_region_path not in accumulator:
                    accumulator[rel_region_path] = {var: {} for var in select_variables}
                region_accumulator = accumulator[rel_region_path]
                for variable_folder, variable_name in select_variables:
                    in_variable_path = os.path.join(
                        region_path, condition, day, variable_folder
                    )
                    continuous_splits = get_continuous_splits(in_variable_path)
                    """
                    for rel_file_path, file_path in listdir(
                        in_variable_path
                    ):
                        file_content = netCDF4.Dataset(file_path)
                        data = np.array(file_content[variable_name][:])
                        tensor_data = t.from_numpy(data)
                        region_accumulator[
                            select_variables.index(
                                (variable_folder, variable_name)
                            )
                        ].append(tensor_data)
                    """
                    region_accumulator[(variable_folder, variable_name)][
                        day
                    ] = continuous_splits
            """
            tensorized_accumulator = (
                nested_tensor_list_to_tensor(accumulator)
                .transpose(1, 2)
                .contiguous()
            ).transpose(0, 1)
            if tensorized_accumulator.shape[1] == n_regions:
                file_name = os.path.join(out_condition_path, f"{day_idx}.pt")
                assert (
                    tensorized_accumulator.shape[1] == n_regions
                ), "wrong shape"
                t.save(tensorized_accumulator, file_name)
            """
        continuous_blocks = split_continuous_blocks_at_root(merge_days(accumulator))
        print("Saving stuff")
        for i, block in tqdm(tuple(enumerate(continuous_blocks))):
            tensor_block = block_to_tensor(block)
            if len(tensor_block) > 9:
                file_name = os.path.join(out_condition_path, f"{i}.pt")
                t.save(tensor_block, file_name)
                metadata[condition]["length"] += len(tensor_block)
            else:
                print("Skipped tensor because it was too small")
                print(len(tensor_block))
        print(f"Done {condition}")
    print("Writing metadata:")
    print(json.dumps(metadata, indent=4))
    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
