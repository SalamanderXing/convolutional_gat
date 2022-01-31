import torch as t
import torch.nn as nn
from torchinfo import summary
import os
from tqdm import tqdm
import json
import ipdb
from argparse import ArgumentParser
from .data_loaders.get_loaders import get_loaders
from .model import SpatialModel, TemporalModel
import matplotlib.pyplot as plt
from .unet_model import UnetModel
import climage

# from .utils import thresholded_mask_metrics, update_history

# todo: add that it saves the best performing model


def plot_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save=False,
    filename="history",
):
    plt.clf()
    plt.plot(
        history["train_loss"],
        label="Train loss",
    )
    plt.plot(
        history["val_loss"],
        label="Val loss",
    )
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def update_history(history: dict[str, list[float]], data: dict[str, float]):
    for key, val in data.items():
        if key not in history:
            history[key] = []
        history[key].append(val)


def get_metrics(y, y_hat, mean):
    y = t.clone(y.cpu())
    y_hat = t.clone(y_hat.cpu())
    y[y < mean] = 0
    y[y >= mean] = 1
    y_hat[y_hat < mean] = 0
    y_hat[y_hat >= mean] = 1
    acc = accuracy(y, y_hat)
    prec = precision(y, y_hat)
    # if prec == t.nan:
    #    ipdb.set_trace()
    rec = recall(y, y_hat)
    return acc, prec, rec


def accuracy(y, y_hat):
    return (y == y_hat).sum() / y[0].numel()


def precision(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    return (TP / (TP + FP)) * len(y_true)


def recall(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    # FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    result = (TP / (TP + FN)) * len(y_true)
    # jif result.isnan():
    #    ipdb.set_trace()
    return result


def denormalize(x, mean, var):
    mean = t.mean(mean)
    var = t.var(var)
    return x * var + mean


def test(model: nn.Module, device, val_test_loader, flag="val"):
    binarize_thresh = t.mean(val_test_loader.normalizing_mean)
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
        mean = val_test_loader.normalizing_mean
        var = val_test_loader.normalizing_var
        threshold = (0.5 - t.mean(val_test_loader.normalizing_mean)) / t.mean(
            val_test_loader.normalizing_var
        )
        for i, (x, y) in tqdm(enumerate(val_test_loader)):
            if len(x) > 1:
                y_hat = model(x)
                running_loss += (
                    t.sum((y - y_hat) ** 2) / t.prod(t.tensor(y.shape[1:]).to(device))
                ).cpu()

                unique = t.unique(y)
                threshold = unique[int(len(unique) * (1 / 4))].cpu()
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
                    t.sum(
                        ((denormalize(y, mean, var) - denormalize(y_hat, mean, var)))
                        ** 2
                    )
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


def term_display(y, y_hat):

    plt.clf()

    if len(y.shape) == 4:
        im1 = y[0, 0, :20, :20].detach().cpu()
        im2 = y_hat[0, 0, :20, :20].detach().cpu()
    else:
        im1 = y[0, 0, 0].detach().cpu()
        im2 = y_hat[0, 0, 0].detach().cpu()

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()
    _fig, ax = plt.subplots(nrows=1, ncols=2)
    ims = [im1, im2]

    for i, col in enumerate(ax):
        col.imshow(ims[i])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("/tmp/im1.png")
    print(
        climage.convert(
            "/tmp/im1.png",
            is_unicode=True,
        )
    )


def visualize_predictions(
    model,
    epoch=1,
    path="",
    downsample_size=(256, 256),
    preprocessed_folder: str = "",
    dataset="kmni",
):
    plt.clf()
    with t.no_grad():
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        merge_nodes = issubclass(UnetModel, type(model))
        loader, _, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=2,
            preprocessed_folder=preprocessed_folder,
            device=device,
            downsample_size=downsample_size,
            dataset=dataset,
            merge_nodes=merge_nodes,
        )
        model.eval()
        N_COLS = 4  # frames
        N_ROWS = 3  # x, y, preds
        plt.title(f"Epoch {epoch}")
        _fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS)
        for x, y in loader:
            for k in range(len(x)):
                raininess = t.sum(x[k] != 0) / t.prod(t.tensor(x[k].shape))
                if raininess >= 0.5:
                    preds = model(x)
                    to_plot = [x[k], y[k], preds[k]]
                    for i, row in enumerate(ax):
                        for j, col in enumerate(row):
                            # ipdb.set_trace()
                            col.imshow(
                                to_plot[i].cpu().detach().numpy()[:, :, j, 1]
                                if not merge_nodes
                                else to_plot[i]
                                .cpu()
                                .detach()
                                .numpy()[
                                    j,
                                    : downsample_size[0],
                                    : downsample_size[1],
                                ]
                            )

                    row_labels = ["x", "y", "preds"]
                    for ax_, row in zip(ax[:, 0], row_labels):
                        ax_.set_ylabel(row)

                    col_labels = ["frame1", "frame2", "frame3", "frame4"]
                    for ax_, col in zip(ax[0, :], col_labels):
                        ax_.set_title(col)

                    plt.savefig(os.path.join(path, f"pred_{epoch}.png"))
                    plt.close()
                    model.train()
                    term_display(y, preds)
                    return


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
    merge_nodes = issubclass(UnetModel, type(model))
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        preprocessed_folder=preprocessed_folder,
        device=device,
        dataset=dataset,
        downsample_size=downsample_size,
        merge_nodes=merge_nodes,
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
            loss = criterion(y_hat, y) - 0.0005 * (t.sum(y_hat) / y_hat.numel())
            loss.backward()  # Update the gradients
            optimizer.step()  # Adjust model parameters
            total_length += len(x)
            running_loss += (
                (t.sum((y_hat - y) ** 2) / t.prod(t.tensor(y.shape[1:]).to(device)))
                .detach()
                .cpu()
            )
    # print(f"{__total_length=}")
    scheduler.step()
    train_loss = (running_loss / total_length).item()
    print(f"Train loss: {round(train_loss, 6)}")
    history["train_loss"].append(train_loss)
    test_result = test(model, device, val_loader)
    # print(f"Val loss: {round(test_result['val_loss'], 6)}")
    print(json.dumps(test_result, indent=4))
    update_history(history, test_result)
    with open(output_path + "/history.json", "w") as f:
        json.dump(history, f, indent=4)
    if len(history["val_loss"]) > 1 and test_result["val_loss"] < min(
        history["val_loss"][:-1]
    ):
        t.save(model.state_dict(), output_path + "/model.pt")


def get_number_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    *,
    model_class,
    optimizer_class,
    mapping_type,
    train_batch_size=1,
    test_batch_size=100,
    epochs=10,
    lr=0.001,
    lr_step=1,
    gamma=1.0,
    plot=True,
    criterion=nn.MSELoss(),
    downsample_size=(256, 256),
    output_path=".",
    preprocessed_folder="",
    dataset="kmni",
    test_first=False,
):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    history = {"train_loss": []}
    print(f"Using device: {device}")
    merge_nodes = model_class == UnetModel
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
    model = model_class(
        image_width=image_width,
        image_height=image_height,
        n_vertices=n_vertices,
        mapping_type=mapping_type,
    ).to(device)
    print(f"Number of parameters: {get_number_parameters(model)}")
    print(f"Using mapping: {model.mapping_type}")

    # summary(model, input_size=x.shape)

    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=gamma)
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
