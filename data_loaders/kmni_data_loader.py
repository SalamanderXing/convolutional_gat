import torch as t
from threading import Thread
import os
import numpy as np
import ipdb
from enum import Enum, unique
from tqdm import tqdm
import json

# todo: shuffling
# todo: fix the fist batch is empty


class DataLoader:
    def __init__(
        self,
        batch_size: int,
        folder: str,
        device,
        *,
        time_steps: int = 4,
        crop=None,
        shuffle: bool = True,
        merge_nodes: bool = False
    ):
        self.merge_nodes = merge_nodes
        self.crop = crop
        self.device = device
        self.batch_size = batch_size
        self.file_index = 0
        self.folder = folder
        self.files = tuple(
            os.path.join(folder, fn) for fn in sorted(os.listdir(folder))
        )
        self.shuffle = shuffle
        if self.shuffle:
            rand_indices = t.randperm(len(self.files))
            tmp = tuple(self.files[i] for i in rand_indices)
            self.files = tmp
        self.remainder = self.__read_next_file()
        self.file_length = self.remainder.shape[0] * self.remainder.shape[1]
        with open(os.path.join(folder, "../metadata.json")) as f:
            metadata = json.load(f)
            self.normalizing_var = metadata["var"]
            self.normalizing_mean = metadata["mean"]

    """
    def __len__(self) -> int:
        return int(
            np.ceil(len(self.files) * self.file_length / self.batch_size)
        )
    """

    def __read_next_file(self) -> t.Tensor:
        if self.file_index == len(self.files):
            raise StopIteration
        data = t.load(self.files[self.file_index])
        self.file_index += 1
        result = self.__segmentify(data)
        return result

    def __segmentify(self, data: t.Tensor) -> t.Tensor:
        data = data[: (len(data) // 8) * 8]
        segments = t.stack(
            tuple(
                el
                for el in tuple(data[i : i + 8] for i in range(len(data)))
                if len(el) == 8
            )
        )
        """
        segments = data.view(
            -1, 8, data.shape[1], data.shape[2], data.shape[3]
        )
        """
        split_segments = t.stack(
            tuple(t.stack((s[:4], s[4:])) for s in segments)
        ).transpose(0, 1)
        if self.crop is not None:
            split_segments = split_segments[:, :, :, :, : self.crop, : self.crop]
        if self.merge_nodes:
            split_segments = t.cat(
                tuple(
                    t.cat(
                        (split_segments[:, :, :, i], split_segments[:, :, :, i + 1],),
                        dim=3,
                    )
                    for i in range(3)
                ),
                dim=4,
            )
        return split_segments

    def __next__(self) -> tuple[t.Tensor, t.Tensor]:
        if self.remainder.shape[1] == 0:
            data = self.__read_next_file()
        else:
            data = self.remainder
        self.remainder = data[:, self.batch_size :]
        result = data[:, : self.batch_size].to(self.device)
        rand_indices = (
            t.randperm(result.shape[1]) if self.shuffle else t.arange(result.shape[1])
        )
        results = (
            (result[0][rand_indices].permute(0, 3, 4, 1, 2) - self.normalizing_mean)
            / self.normalizing_var,
            (result[1][rand_indices].permute(0, 3, 4, 1, 2) - self.normalizing_mean)
            / self.normalizing_var,
        )
        return results

    def __iter__(self):
        return self


def get_loaders(
    train_batch_size: int,
    test_batch_size: int,
    data_folder: str,
    device,
    crop: int = None,
    shuffle: bool = True,
):
    train_loader = DataLoader(
        train_batch_size,
        os.path.join(data_folder, "train"),
        device,
        crop=crop,
        shuffle=shuffle,
    )

    val_loader = DataLoader(
        test_batch_size,
        os.path.join(data_folder, "test"),
        device,
        crop=crop,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        test_batch_size,
        os.path.join(data_folder, "test"),
        device,
        crop=crop,
        shuffle=shuffle,
    )
    return train_loader, val_loader, test_loader


def test():
    data_loader = DataLoader(
        batch_size=32,
        folder="/mnt/kmni_dataset/preprocessed/",
        device=t.device("cuda" if t.cuda.is_available() else "cpu"),
    )
    # print(f"{len(data_loader)}")
    for x, y in tqdm(data_loader):
        pass


if __name__ == "__main__":
    test()
