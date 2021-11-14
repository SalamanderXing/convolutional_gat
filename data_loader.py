import torch as t
from threading import Thread
import os
import numpy as np
import ipdb
from enum import Enum, unique

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
        task: Task = Task.predict_next,
        time_steps: int = 4,
        norm_max=None,
        norm_min=None,
    ):
        self.foder = folder
        self.task = task
        self.device = device
        self.__is_first = True
        self.norm_max = norm_max
        self.norm_min = norm_min

        self.time_steps = time_steps
        self.__batch_size = batch_size
        if self.task == task.predict_next:
            self.__batch_size *= 2  # taken into account the time step of 1
            self.__batch_size += self.time_steps - 1
        self.__next_batch = t.tensor([])
        self.__remainder = t.tensor([])
        self.file_index = 0
        self.files = [f for f in os.listdir(folder)]
        self.item_count = 24 * 4 * max(int(f.split(".")[0]) for f in self.files)

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
        return int(np.ceil(self.item_count / self.__batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.__is_first:
            self.__get_batch()
        while self.__next_batch == None:
            pass
        current_batch = self.__next_batch
        self.__next_batch = None
        thread = Thread(target=self.__get_batch)
        thread.start()
        return self.__batchify(current_batch)

    def __read_next_file(self):
        if self.file_index == len(self.files):
            raise StopIteration
        tensor = t.load(os.path.join(self.foder, f"{self.files[self.file_index]}"))
        self.file_index += 1
        return tensor

    def __get_batch(self):
        accumulator = self.__remainder
        while len(accumulator) < self.__batch_size:
            to_be_gained = self.__batch_size - accumulator.shape[0]
            next_batch = self.__read_next_file()
            new_data = next_batch[:to_be_gained]
            accumulator = (
                new_data if len(accumulator) == 0 else t.cat((accumulator, new_data))
            )
            self.__remainder = next_batch[to_be_gained:]
        self.__next_batch = accumulator


def get_loaders(
    train_batch_size: int,
    test_batch_size: int,
    preprocessed_folder: str,
    device,
    task: Task,
):
    return (
        DataLoader(
            train_batch_size,
            os.path.join(preprocessed_folder, "training"),
            device,
            task,
        ),
        DataLoader(
            test_batch_size,
            os.path.join(preprocessed_folder, "validation"),
            device,
            task,
        ),
        DataLoader(
            test_batch_size, os.path.join(preprocessed_folder, "test"), device, task,
        ),
    )


def main():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=32,
        test_batch_size=100,
        preprocessed_folder="preprocessed",
        device=device,
        task=Task.predict_next,
    )
    for batch in train_loader:
        ipdb.set_trace()


if __name__ == "__main__":
    main()
