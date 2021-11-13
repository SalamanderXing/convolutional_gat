import torch as t
import torch.nn as nn
from tqdm import tqdm
import ipdb
from argparse import ArgumentParser
from .data_loader import get_loaders
from .model import ConvGAT


# todo: add that it saves the best performing model


def plot_history(history):
    pass


def test(model, device, val_test_set):
    model.eval()  # We put the model in eval mode: this disables dropout for example (which we didn't use)
    with t.no_grad():  # Disables the autograd engine
        running_loss = 0.0
        total_length = 0
        for data in tqdm(val_test_set):
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            running_loss += t.sum((inputs - outputs) ** 2).item()
            total_length += len(inputs)
    model.train()
    return running_loss / total_length


def train(
    model_class=ConvGAT,
    train_batch_size=32,
    test_batch_size=100,
    epochs=10,
    lr=0.001,
    lr_step=1,
    gamma=1.0,  # 1.0 means disabled
    plot=True,
):
    device = t.device(
        "cuda" if t.cuda.is_available() else "cpu"
    )  # Select the GPU device, if there is one available.
    #
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        preprocessed_folder="convolutional-gat/preprocessed",
        device=device,
    )
    print(
        f"Using: {device}\n\nSizes:\n train: {train_loader.item_count}\n val: {val_loader.item_count}\n test: {test_loader.item_count}\n"
    )
    model = model_class().to(device)  # The model always stays on the GPU
    # optimizer = the procedure for updating the weights of our neural network
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()  tested but didn't improve significantly
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=gamma)
    history = []
    for epoch in range(epochs):
        model.train()
        print(f"\nEpoch: {epoch + 1}")
        running_loss = 0.0
        total_length = 0
        for param_group in optimizer.param_groups:  # Print the updated LR
            print(f"LR: {param_group['lr']}")
        for data in tqdm(train_loader):
            inputs, _ = data
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            inputs = inputs.to(
                device
            )  # We move the tensors to the GPU for (much) faster computation
            outputs = model(inputs)  # Implicitly calls the model's forward function
            loss = criterion(outputs, inputs)
            loss.backward()  # Update the gradients
            optimizer.step()  # Adjust model parameters
            total_length += len(inputs)
            running_loss += t.sum((inputs - outputs) ** 2).item()

        scheduler.step()
        train_loss = running_loss / total_length
        print(f"Train loss: {round(train_loss, 6)}")
        val_loss = test(model, device, val_loader)
        print(f"Val loss: {round(val_loss, 6)}")
        history.append((train_loss, val_loss))
    test_loss = test(model, device, test_loader)
    print(f"Test loss: {round(test_loss, 6)}")
    if plot:
        plot_history(history)
    return history, test_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    train()
