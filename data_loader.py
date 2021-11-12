import torch as t
from threading import Thread
import os
import numpy as np
import ipdb

# todo: shuffling
# todo: fix the fist batch is empty


class DataLoader:
    def __init__(self, batch_size: int, folder: str, device):
        self.foder = folder
        self.batch_size = batch_size
        self.next_batch = t.tensor([])
        self.remainder = t.tensor([])
        self.file_index = 0
        self.files = [f for f in os.listdir(folder)]
        self.item_count = max(int(f.split(".")[0]) for f in self.files)
        self.device = device

    def __len__(self):
        return int(np.ceil(self.item_count / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        while self.next_batch == None:
            pass
        current_batch = self.next_batch
        self.next_batch = None
        thread = Thread(target=self.get_batch)
        thread.start()
        return current_batch.to(self.device)

    def read_next_file(self):
        if self.file_index == len(self.files):
            raise StopIteration
        tensor = t.load(
            os.path.join(self.foder, f"{self.files[self.file_index]}")
        )
        self.file_index += 1
        return tensor

    def get_batch(self):
        accumulator = t.tensor([])
        while len(accumulator) < self.batch_size:
            to_be_gained = self.batch_size - accumulator.shape[0]
            next_batch = self.read_next_file()
            new_data = next_batch[:to_be_gained]
            accumulator = (
                new_data
                if len(accumulator) == 0
                else t.cat((accumulator, new_data))
            )
            self.remainder = next_batch[to_be_gained:]
        self.next_batch = accumulator


def get_loaders(
    train_batch_size: int,
    test_batch_size: int,
    preprocessed_folder: str,
    device,
):
    return (
        DataLoader(
            train_batch_size,
            os.path.join(preprocessed_folder, "training"),
            device,
        ),
        DataLoader(
            test_batch_size,
            os.path.join(preprocessed_folder, "validation"),
            device,
        ),
        DataLoader(
            test_batch_size, os.path.join(preprocessed_folder, "test"), device
        ),
    )


def main():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=32,
        test_batch_size=100,
        preprocessed_folder="preprocessed",
        device=device,
    )
    for batch in train_loader:
        print(batch.shape)
        ipdb.set_trace()


if __name__ == "__main__":
    main()
