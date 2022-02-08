import torch as t
import torch.nn as nn
from torchinfo import summary
import os
from tqdm import tqdm
import json
import ipdb
from argparse import ArgumentParser
from .data_loaders.get_loaders import get_loaders
from .GAT3D.GATMultistream import Model
import matplotlib.pyplot as plt
from .unet_model import UnetModel
from .utils import (
    get_metrics,
    visualize_predictions,
    update_history,
    denormalize,
    plot_history,
    model_classes,
    get_number_parameters,
)

# from .utils import thresholded_mask_metrics, update_history

# todo: add that it saves the best performing model


def test(model: nn.Module, device, loader, flag="val"):
    # binarize_thresh = t.mean(val_test_loader.normalizing_mean)
    """
    thresh_metrics = thresholded_mask_metrics(
        threshold=binarize_thresh,
        var=t.mean(val_test_loader.normalizing_var),
        mean=t.mean(val_test_loader.normalizing_mean),
    )
    """
    # val_test_loader.stats(model)
    model.eval()  # We put the model in eval mode: this disables dropout for example (which we didn't use)
    with t.no_grad():  # Disables the autograd engine
        running_loss = t.tensor(0.0)
        running_acc = t.tensor(0.0)
        running_prec = t.tensor(0.0)
        running_recall = t.tensor(0.0)
        running_denorm_mse = t.tensor(0.0)
        total_length = 0
        # mean = val_test_loader.normalizing_mean
        # var = val_test_loader.normalizing_var
        # threshold = (0.5 - t.mean(val_test_loader.normalizing_mean)) / t.mean(
        #    val_test_loader.normalizing_var
        # )
        for i, (x, y) in tqdm(enumerate(loader)):
            if len(x) > 1:
                y_hat = model(x)
                y = t.pow(y, 1 / loader.power)
                y_hat = t.pow(y_hat, 1 / loader.power)
                running_loss += (
                    t.sum((y - y_hat) ** 2)
                    / t.prod(t.tensor(y.shape[1:]).to(device))
                ).cpu()

                unique = t.unique(y)
                threshold = unique[int(len(unique) * (1 / 2))].cpu()
                total_length += len(x)
                acc, prec, rec = get_metrics(
                    y.detach(),
                    y_hat.detach(),
                    threshold,  # second_min  # 0.04011
                )
                running_acc += acc
                running_prec += prec if not prec.isnan() else 0
                running_recall += rec if not rec.isnan() else 0

                running_denorm_mse += (
                    t.sum(((y - y_hat) * loader.normalizing_max) ** 2)
                    / t.prod(t.tensor(y.shape[1:]).to(device))
                ).cpu()
                # running_acc += thresh_metrics.acc(y, y_hat).numpy()
                """
                running_prec += thresh_metrics.precision(
                    y, y_hat
                ).numpy() * len(x)
                running_recall += thresh_metrics.recall(
                    y, y_hat
                ).numpy() * len(x)
                """

    model.train()
    return {
        "val_loss": (running_loss / total_length).item(),
        "val_acc": (running_acc / total_length).item(),
        "val_prec": (running_prec / total_length).item(),
        "val_rec": (running_recall / total_length).item(),
        "val_denorm_mse": (running_denorm_mse / total_length).item(),
    }


def train_single_epoch(
    epoch: int,
    optimizer,
    criterion,
    scheduler,
    model,
    train_batch_size,
    test_batch_size,
    preprocessed_folder,
    device,
    dataset,
    downsample_size,
    history,
    output_path,
):
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        preprocessed_folder=preprocessed_folder,
        device=device,
        dataset=dataset,
        downsample_size=downsample_size,
        merge_nodes=False,
    )
    model.train()
    print(f"\nEpoch: {epoch}")
    running_loss = t.tensor(0.0)
    total_length = 0
    __total_length = 0
    for param_group in optimizer.param_groups:  # Print the updated LR
        print(f"LR: {param_group['lr']}")
    for x, y in tqdm(train_loader):
        __total_length += len(x)
        if len(x) > 1:
            # N(batch size), H,W(feature number) = 256,256, T(time steps) = 4, V(vertices, # of cities) = 5
            optimizer.zero_grad()
            y_hat = model(x)  # Implicitly calls the model's forward function
            loss = criterion(y_hat, y) - 0.0005 * (
                t.sum(y_hat) / y_hat.numel()
            )
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
    # print(f"{__total_length=}")
    train_loss = (running_loss / total_length).item()
    print(f"Train loss: {round(train_loss, 6)}")
    history["train_loss"].append(train_loss)
    test_result = test(model, device, val_loader)
    scheduler.step(test_result["val_loss"])
    # print(f"Val loss: {round(test_result['val_loss'], 6)}")
    print(json.dumps(test_result, indent=4))
    update_history(history, test_result)
    with open(os.path.join(output_path, "history.json"), "w") as f:
        json.dump(history, f, indent=4)
    if (len(history["val_loss"]) == 1) or test_result["val_loss"] < min(
        history["val_loss"][:-1]
    ):
        print("Saving model.")
        t.save(model.state_dict(), os.path.join(output_path, "model.pt"))


def train(
    *,
    model_type,
    optimizer,
    mapping_type,
    output_path,
    train_batch_size,
    test_batch_size,
    epochs,
    learning_rate,
    lr_step,
    gamma,
    plot=True,
    criterion=nn.MSELoss(),
    downsample_size=(256, 256),
    preprocessed_folder="",
    dataset="kmni",
    test_first=False,
    reduce_lr_on_plateau=False,
):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    history = {"train_loss": []}
    print(f"Using device: {device}")
    merge_nodes = False
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        preprocessed_folder=preprocessed_folder,
        device=device,
        dataset=dataset,
        downsample_size=downsample_size,
        merge_nodes=merge_nodes,
    )
    for x, y in val_loader:
        if not merge_nodes:
            _, image_width, image_height, steps, n_vertices = x.shape
        else:
            _, image_width, image_height, _ = x.shape
            n_vertices = 6  # unused in this case
        break
    model_class = model_classes[model_type]
    model = model_class(
        image_width=image_width,
        image_height=image_height,
        n_vertices=n_vertices,
        attention_type=model_type,
        mapping_type=mapping_type,
    ).to(device)

    print(f"Number of parameters: {get_number_parameters(model)}")
    print(f"Using mapping: {model.mapping_type}")

    # summary(model, input_size=x.shape)

    optimizer = optimizer(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    if not reduce_lr_on_plateau:
        scheduler = t.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step, gamma=gamma
        )
    else:
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=0, verbose=True, factor=0.5
        )

    if test_first:
        result = test(
            model,
            device,
            train_loader,
        )
        history["train_loss"].append(result["val_loss"])
        result = test(
            model,
            device,
            test_loader,
        )
        print(f"Test loss (without any training): {result['val_loss']:.6f}")
        update_history(history, result)
        print(json.dumps(result, indent=4))

    for epoch in range(1, epochs + 1):
        train_single_epoch(
            epoch,
            optimizer,
            criterion,
            scheduler,
            model,
            train_batch_size,
            test_batch_size,
            preprocessed_folder,
            device,
            dataset,
            downsample_size,
            history,
            output_path,
        )
        visualize_predictions(
            model,
            epoch=epoch,
            path=output_path,
            downsample_size=downsample_size,
            preprocessed_folder=preprocessed_folder,
            dataset=dataset,
        )
        plot_history(
            history,
            title="Training History",
            save=True,
            filename=os.path.join(output_path, f"history_{epoch}.png"),
        )
    # test_loss = test(model, device, test_loader, "test")
    # print(f"Test loss: {round(test_loss['val_loss'], 6)}")

    # if plot:
    #   plot_history(history)
    return history  # , test_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    train()
