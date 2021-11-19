import torch as t
from threading import Thread
import os
import numpy as np
import ipdb
from enum import Enum, unique
from tqdm import tqdm

# todo: shuffling
# todo: fix the fist batch is empty


@unique
class Task(Enum):
    predict_next = "predict_next"


class DataLoader:
    def __init__(
        self,
        batch_size: int,
        folder: str,
        device,
        *,
        task: Task = Task.predict_next,
        time_steps: int = 4,
        norm_max=None,
        norm_min=None,
        downsample_size: tuple[int, int] = (
            256,
            256,
        ),  # by default, don't downsample
    ):
        self.downsample_size = downsample_size
        self.folder = folder
        self.task = task
        self.device = device
        self.__is_first = True
        self.norm_max = norm_max
        self.norm_min = norm_min
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.__batch_size = batch_size
        if self.task == task.predict_next:
            self.__batch_size *= 2  # taken into account the time step of 1
            self.__batch_size += self.time_steps - 1
        self.__next_batch = t.tensor([])
        self.__remainder = t.tensor([])
        self.file_index = 0
        self.should_stop_iteration = False
        self.files = sorted(
            [f for f in os.listdir(folder)], key=lambda x: int(x.split(".")[0])
        )
        max_file = max(int(f.split(".")[0]) for f in self.files)
        print(f"{max_file=}")
        self.item_count = 86 * len(self.files)
        self.thread = Thread(target=self.__get_batch)
        print(f"{self.files=}")
        print(f"{self.item_count=}")

    def __batchify(self, data) -> tuple[t.Tensor, t.Tensor]:
        result = (t.tensor([]), t.tensor([]))
        if self.task == Task.predict_next:
            shifted = t.stack(
                tuple(
                    data[i : i + self.time_steps]
                    for i in range(len(data) - (self.time_steps - 1))
                )
            )
            even_mask = t.arange(len(shifted)) % 2 == 0
            labels = shifted[even_mask]
            xs = shifted[t.logical_not(even_mask)]
            result = (xs, labels)
        return (result[0].to(self.device), result[0].to(self.device))

    def __len__(self):
        return (2 * self.item_count - self.time_steps + 1) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.thread.is_alive():
            self.thread.join()
        if self.__is_first:
            self.__is_first = False
            self.__get_batch()
        if self.should_stop_iteration:
            raise StopIteration
        current_batch = self.__next_batch
        self.__next_batch = None
        try:
            self.thread.start()
        except:
            self.thread = Thread(target=self.__get_batch)
            self.thread.start()
        return self.__batchify(current_batch)

    def __read_next_file(self):
        if self.file_index == len(self.files):
            self.should_stop_iteration = True
        tensor = t.tensor([[]])
        # while (
        #    tensor.shape[1] < 5
        # ):  # TODO: some files have apparently only 5 in the second dimension, could mean there is a bug in the preprocessing or the data is not perfect
        tensor = t.load(
            os.path.join(self.folder, f"{self.files[self.file_index]}")
        )
        if len(tensor.shape) > 5:
            ipdb.set_trace()
        tensor = tensor[
            :, :, :, : self.downsample_size[0], : self.downsample_size[1]
        ]
        # print(f"{tensor.shape=}")
        if tensor.shape[1] < 5:
            print("skipping")

        # print(self.file_index)
        # print(self.files[self.file_index])
        self.file_index += 1
        if self.file_index == len(self.files):
            self.should_stop_iteration = True

        # TODO: apply downsampling_factor
        return tensor

    def __get_batch(self):
        accumulator = self.__remainder
        self.__remainder = t.tensor([])
        while len(accumulator) < self.__batch_size:
            to_be_gained = self.__batch_size - len(accumulator)
            next_batch = self.__read_next_file()
            new_data = next_batch[:to_be_gained]
            # print(f"{new_data.shape=}")
            # print(f"{accumulator.shape=}")
            accumulator = (
                new_data
                if len(accumulator) == 0
                else t.cat((accumulator, new_data))
            )
            self.__remainder = next_batch[to_be_gained:]
        self.__next_batch = accumulator


def get_loaders(
    train_batch_size: int,
    test_batch_size: int,
    preprocessed_folder: str,
    device,
    *,
    task: Task,
    downsample_size: tuple[int, int] = (256, 256),
):
    return (
        DataLoader(
            train_batch_size,
            os.path.join(preprocessed_folder, "training"),
            device,
            task=task,
            downsample_size=downsample_size,
        ),
        DataLoader(
            test_batch_size,
            os.path.join(preprocessed_folder, "validation"),
            device,
            task=task,
            downsample_size=downsample_size,
        ),
        DataLoader(
            test_batch_size,
            os.path.join(preprocessed_folder, "test"),
            device,
            task=task,
            downsample_size=downsample_size,
        ),
    )


def test():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=32,
        test_batch_size=100,
        preprocessed_folder="preprocessed",
        device=device,
        task=Task.predict_next,
        downsample_size=(16, 16),
    )
    print(f"{len(train_loader)=}")
    i = 0
    total_length = 0
    for x, y in tqdm(train_loader):
        print(f"{x.shape=}")
        print(f"{y.shape=}")
        total_length += len(x)
        i += 1
    print(f"{total_length=}")
    print(f"Iterated {i} times")


if __name__ == "__main__":
    test()