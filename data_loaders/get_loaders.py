import os
import json
from .arai_data_loader import get_loaders as get_loaders_arai
from .kmni_data_loader import get_loaders as get_loaders_kmni


def get_loaders(
    train_batch_size: int,
    test_batch_size: int,
    preprocessed_folder: str,
    device,
    *,
    dataset: str = "kmni",
    downsample_size: tuple[int, int] = (256, 256),
    merge_nodes: bool = False,
    shuffle=True
):
    if dataset == "arai":
        return get_loaders_arai(
            train_batch_size,
            test_batch_size,
            preprocessed_folder,
            device,
            downsample_size=downsample_size,
        )
    elif dataset == "kmni":
        return get_loaders_kmni(
            train_batch_size,
            test_batch_size,
            preprocessed_folder,
            device,
            crop=downsample_size[0],
            merge_nodes=merge_nodes,
            shuffle=shuffle,
        )
