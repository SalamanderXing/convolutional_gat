import torch as t
from threading import Thread
import os
import matplotlib.pyplot as plt
import numpy as np
import ipdb
from enum import Enum, unique
from tqdm import tqdm
import json
from ..preprocessing.utils import listdir

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
        merge_nodes: bool = False,
        power: float = 1.0
    ):
        self.power = t.tensor(power)
        # metadata = t.load(os.path.join(folder, "../metadata.pt"))
        self.data_folder = folder
        self.normalizing_max = 254
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

    def stats(self):
        # all_training =
        flat = t.cat(
            tuple(t.load(fp) for fn, fp in listdir(self.data_folder))
        ).view(-1)
        bins = np.unique(flat)
        """
        hist, _ = np.histogram(flat, bins)
        plt.plot(np.arange(hist), hist)
        """
        norm = (flat - t.mean(self.normalizing_mean)) / t.mean(
            self.normalizing_var
        )
        hist, _ = np.histogram(norm, len(bins))
        plt.plot(np.arange(len(hist)), hist)

        plt.show()
        ipdb.set_trace()
        # plt.hist(flat, bins="auto")
        # ipdb.set_trace()

    def __read_next_file(self) -> t.Tensor:
        if self.file_index == len(self.files):
            raise StopIteration
        data = t.load(self.files[self.file_index])
        self.file_index += 1
        result = self.__segmentify(data)
        return result

    def __segmentify(self, data: t.Tensor) -> t.Tensor:
        data = data[: (len(data) // 8) * 8]
        norm_data = data / self.normalizing_max
        data = t.pow(norm_data, self.power)
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
            split_segments = split_segments[
                :, :, :, :, : self.crop, : self.crop
            ]
        if self.merge_nodes:
            split_segments = t.cat(
                tuple(
                    t.cat(
                        (
                            split_segments[:, :, :, i],
                            split_segments[:, :, :, i + 1],
                        ),
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
            t.randperm(result.shape[1])
            if self.shuffle
            else t.arange(result.shape[1])
        )
        if self.merge_nodes:
            result = result.permute(0, 1, 2, 3, 4)
        else:
            result = result.permute(0, 1, 4, 5, 2, 3)
        results = (
            result[0][rand_indices],
            result[1][rand_indices],
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
    merge_nodes: bool = False,
):
    train_loader = DataLoader(
        train_batch_size,
        os.path.join(data_folder, "train"),
        device,
        crop=crop,
        shuffle=shuffle,
        merge_nodes=merge_nodes,
    )

    val_loader = DataLoader(
        test_batch_size,
        os.path.join(data_folder, "test"),
        device,
        crop=crop,
        shuffle=shuffle,
        merge_nodes=merge_nodes,
    )
    test_loader = DataLoader(
        test_batch_size,
        os.path.join(data_folder, "test"),
        device,
        crop=crop,
        shuffle=shuffle,
        merge_nodes=merge_nodes,
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
