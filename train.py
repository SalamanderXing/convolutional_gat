import torch as t
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm
import json
import ipdb
from argparse import ArgumentParser
from .data_loader import get_loaders
from .model import SpatialModel, TemporalModel
import matplotlib.pyplot as plt
from .data_loader import Task

# todo: add that it saves the best performing model


def plot_history(
    history: dict[str, list[float]], title: str = "Training History"
):
    plt.plot(
        t.arange(len(history["train_loss"])),
        history["train_loss"],
        label="Train loss",
    )
    plt.plot(
        t.arange(len(history["train_loss"])),
        history["val_loss"],
        label="Val loss",
    )
    plt.legend()
    plt.title(title)
    plt.show()


def test(model: nn.Module, device, val_test_loader, label="val"):
    model.eval()  # We put the model in eval mode: this disables dropout for example (which we didn't use)
    print(f"{label=}")
    with t.no_grad():  # Disables the autograd engine
        running_loss = 0.0
        total_length = 0
        for data in tqdm(val_test_loader):
            current_time_step, next_time_step = fix_sizes(*data)
            inputs = current_time_step.to(device)
            outputs = model(inputs)
            running_loss += t.sum((next_time_step - outputs) ** 2).item()
            total_length += len(inputs)
    model.train()
    return running_loss / total_length


def fix_sizes(tensor1: t.Tensor, tensor2: t.Tensor):
    tensor1 = tensor1.squeeze(3)  # same
    # print(current_time_step.shape)
    tensor1 = tensor1.permute(0, 3, 4, 1, 2)
    tensor2 = tensor2.squeeze(3)  # same
    # print(current_time_step.shape)
    tensor2 = tensor2.permute(0, 3, 4, 1, 2)
    return tensor1, tensor2


def train(
    model,
    train_batch_size=1,
    test_batch_size=100,
    epochs=10,
    lr=0.001,
    task=Task.predict_next,
    lr_step=1,
    gamma=1.0,  # 1.0 means disabled
    plot=True,
    criterion=nn.MSELoss(),
    optimizer=None,
    downsample_size=(50, 50),
):
    device = t.device(
        "cuda" if t.cuda.is_available() else "cpu"
    )  # Select the GPU device, if there is one available.
    #
    # device = t.device('cpu')
    model = model.to(device)
    summary(
        model,
        input_size=(
            train_batch_size,
            downsample_size[0],
            downsample_size[1],
            4,
            5,
        ),
        device=device,
    )
    # optimizer = the procedure for updating the weights of our neural network
    # optimizer = t.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()  tested but didn't improve significantly
    scheduler = t.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=gamma
    )
    history = {"train_loss": [], "val_loss": []}
    print(f"Using device: {device}")
    for epoch in range(epochs):
        train_loader, val_loader, test_loader = get_loaders(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            preprocessed_folder="convolutional_gat/preprocessed",
            device=device,
            task=task,
            downsample_size=downsample_size,
        )
        # print(
        #    f"Using: {device}\n\nSizes:\n train: {train_loader.item_count}\n val: {val_loader.item_count}\n test: {test_loader.item_count}\n"
        # )

        model.train()
        print(f"\nEpoch: {epoch + 1}")
        running_loss = 0.0
        total_length = 0
        for param_group in optimizer.param_groups:  # Print the updated LR
            print(f"LR: {param_group['lr']}")
        for current_time_step, next_time_step in tqdm(train_loader):
            # N(batch size), H,W(feature number) = 256,256, T(time steps) = 4, V(vertices, # of cities) = 5
            current_time_step, next_time_step = fix_sizes(
                current_time_step, next_time_step
            )
            optimizer.zero_grad()
            predicted_next_time_step = model(
                current_time_step
            )  # Implicitly calls the model's forward function
            loss = criterion(predicted_next_time_step, next_time_step)
            loss.backward()  # Update the gradients
            optimizer.step()  # Adjust model parameters
            total_length += len(current_time_step)
            running_loss += t.sum(
                (predicted_next_time_step - next_time_step) ** 2
            ).item()

        scheduler.step()
        train_loss = running_loss / total_length
        print(f"Train loss: {round(train_loss, 6)}")
        val_loss = test(model, device, val_loader)
        print(f"Val loss: {round(val_loss, 6)}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        with open("history.json", "w") as f:
            json.dump(history, f)
        if val_loss < min(history["val_loss"]):
            t.save(model.state_dict(), "model.pt")
    test_loss = test(model, device, test_loader, "test")
    print(f"Test loss: {round(test_loss, 6)}")
    if plot:
        plot_history(history)
    return history, test_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    train()
