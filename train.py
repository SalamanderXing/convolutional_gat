import torch as t
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm
import json
import ipdb
from argparse import ArgumentParser
from .data_loaders.get_loaders import get_loaders
from .model import SpatialModel, TemporalModel
import matplotlib.pyplot as plt

# todo: add that it saves the best performing model


def plot_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save=False,
    filename="train",
):
    plt.plot(
        history["train_loss"], label="Train loss",
    )
    plt.plot(
        history["val_loss"], label="Val loss",
    )
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig(filename)
    else:
        plt.show()


def test(model: nn.Module, device, val_test_loader, label="val"):
    model.eval()  # We put the model in eval mode: this disables dropout for example (which we didn't use)
    with t.no_grad():  # Disables the autograd engine
        running_loss = t.tensor(0.0)
        total_length = 0
        for x, y in tqdm(val_test_loader):
            if len(x) > 1:
                y_hat = model(x)
                running_loss += (
                    t.sum((y - y_hat) ** 2)
                    / t.prod(t.tensor(y.shape[1:]).to(device))
                ).cpu()
                total_length += len(x)
    model.train()
    return (running_loss / total_length).item()


def visualize_predictions(
    model,
    number_of_preds=1,
    path="",
    downsample_size=(256, 256),
    preprocessed_folder: str = "",
    dataset="kmni",
):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    loader, _, _ = get_loaders(
        train_batch_size=1,
        test_batch_size=1,
        preprocessed_folder=preprocessed_folder,
        device=device,
        downsample_size=downsample_size,
        dataset=dataset,
    )
    model = model.to(device)
    N_COLS = 4  # frames
    N_ROWS = 3  # x, y, preds
    _fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS)
    for x, y in loader:
        x, y = x[:number_of_preds], y[:number_of_preds]
        preds = model(x)

        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                if i == 0:
                    col.imshow(x.cpu().detach().numpy().squeeze(0)[:, :, j, 0])
                elif i == 1:
                    col.imshow(y.cpu().detach().numpy().squeeze(0)[:, :, j, 0])
                else:
                    col.imshow(
                        1 - preds.cpu().detach().numpy().squeeze(0)[:, :, j, 0]
                    )

        row_labels = ["x", "y", "preds"]
        for ax_, row in zip(ax[:, 0], row_labels):
            ax_.set_ylabel(row)

        col_labels = ["frame1", "frame2", "frame3", "frame4"]
        for ax_, col in zip(ax[0, :], col_labels):
            ax_.set_title(col)

        plt.savefig(path + "/results_viz.png")
        break


def train(
    model,
    train_batch_size=1,
    test_batch_size=100,
    epochs=10,
    lr=0.001,
    lr_step=1,
    gamma=1.0,  # 1.0 means disabled
    plot=True,
    criterion=nn.MSELoss(),
    optimizer=None,
    downsample_size=(256, 256),
    output_path=".",
    preprocessed_folder="",
    dataset="kmni",
    conv=False,
):
    device = t.device(
        "cuda" if t.cuda.is_available() else "cpu"
    )  # Select the GPU device, if there is one available.
    #
    # device = t.device('cpu')
    model = model.to(device)
    # optimizer = the procedure for updating the weights of our neural network
    # optimizer = t.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()  tested but didn't improve significantly
    scheduler = t.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=gamma
    )
    history = {"train_loss": [], "val_loss": []}
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        preprocessed_folder=preprocessed_folder,
        device=device,
        dataset=dataset,
        downsample_size=downsample_size,
    )
    test_loss = test(model, device, test_loader, "test")
    history["val_loss"].append(test_loss)
    history["train_loss"].append(1.0)
    print(f"Test loss (without any training): {test_loss}")
    for epoch in range(epochs):
        train_loader, val_loader, test_loader = get_loaders(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            preprocessed_folder=preprocessed_folder,
            device=device,
            dataset=dataset,
            downsample_size=downsample_size,
        )
        # print(
        #    f"Using: {device}\n\nSizes:\n train: {train_loader.item_count}\n val: {val_loader.item_count}\n test: {test_loader.item_count}\n"
        # )

        model.train()
        print(f"\nEpoch: {epoch + 1}")
        running_loss = t.tensor(0.0)
        total_length = 0
        for param_group in optimizer.param_groups:  # Print the updated LR
            print(f"LR: {param_group['lr']}")
        for x, y in tqdm(train_loader):
            if len(x) > 1:
                # N(batch size), H,W(feature number) = 256,256, T(time steps) = 4, V(vertices, # of cities) = 5
                optimizer.zero_grad()
                y_hat = model(
                    x
                )  # Implicitly calls the model's forward function
                loss = criterion(y_hat, y)
                loss.backward()  # Update the gradients
                optimizer.step()  # Adjust model parameters
                total_length += len(x)
                running_loss += (
                    (
                        t.sum((y_hat - y) ** 2)
                        / t.prod(t.tensor(y.shape[1:]).to(device))
                    )
                    .detach()
                    .cpu()
                )

        scheduler.step()
        train_loss = (running_loss / total_length).item()
        print(f"Train loss: {round(train_loss, 6)}")
        val_loss = test(model, device, val_loader)
        print(f"Val loss: {round(val_loss, 6)}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        with open(output_path + "/history.json", "w") as f:
            json.dump(history, f)
        if len(history["val_loss"]) > 1 and val_loss < min(
            history["val_loss"][:-1]
        ):
            t.save(model.state_dict(), output_path + "/model.pt")
    test_loss = test(model, device, test_loader, "test")
    print(f"Test loss: {round(test_loss, 6)}")

    plot_history(
        history,
        title="Training History",
        save=True,
        filename=output_path + "/train.png",
    )
    visualize_predictions(
        model,
        number_of_preds=1,
        path=output_path,
        downsample_size=downsample_size,
        preprocessed_folder=preprocessed_folder,
        dataset=dataset,
    )
    if plot:
        plot_history(history)
    return history, test_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    train()
