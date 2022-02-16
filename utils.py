import h5py
import numpy as np
import torch as t
import ipdb
import enum
import matplotlib.pyplot as plt
import climage
from .unet_model import UnetModel
from .GAT3D.GATMultistream import Model as GatModel
from .data_loaders.get_loaders import get_loaders
import os

model_classes = {
    "unet": UnetModel,
    "temporal": GatModel,
    "spatial": GatModel,
    "multi_stream": GatModel,
}


def get_number_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        train_loader, test_loader, _ = get_loaders(
            train_batch_size=2,
            test_batch_size=2,
            preprocessed_folder=preprocessed_folder,
            device=device,
            downsample_size=downsample_size,
            dataset=dataset,
            merge_nodes=False,
            shuffle=True,
        )
        model.eval()
        N_COLS = 4  # frames
        N_ROWS = 3  # x, y, preds
        plt.title(f"Epoch {epoch}")
        _fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS)
        for x, y in test_loader:
            for k in range(len(x)):
                raininess = t.sum(x[k] > 0.0) / t.prod(t.tensor(x[k].shape))
                if raininess >= 0.5:
                    preds = model(x)
                    to_plot = [
                        t.pow(val, 1 / test_loader.power)
                        for val in [x[k], y[k], preds[k]]
                    ]
                    for i, row in enumerate(ax):
                        for j, col in enumerate(row):
                            # ipdb.set_trace()
                            col.imshow(
                                to_plot[i].cpu().detach().numpy()[:, :, j, 1]
                            )

                    row_labels = ["x", "y", "preds"]
                    for ax_, row in zip(ax[:, 0], row_labels):
                        ax_.set_ylabel(row)

                    col_labels = ["frame1", "frame2", "frame3", "frame4"]
                    for ax_, col in zip(ax[0, :], col_labels):
                        ax_.set_title(col)

                    save_path = os.path.join(path, f"pred_{epoch}.png")
                    plt.savefig(save_path)
                    plt.close()
                    model.train()
                    # term_display(y, preds)
                    return
    print("Raininess threshold too strict, hasn't found anything")


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


"""
# Custom metric to calculate the denormalized for the precipitation data
class MSE_denormalized:
    def __init__(self, loader, reduction_sum=False):
        # self.maxValue = maxValue
        self.batch_size = loader.batch_size
        # self.n_pixels = t.multiply(latitude, longitude), tf.float32
        self.var = loader.normalizing_var
        self.mean = loader.normalizing_mean
        if reduction_sum:
            self.mse = t.nn.MSELoss(reduction="sum")
        else:
            self.mse = t.nn.MSELoss()

    def mse_denormalized_per_image(self, y_true, y_pred):
        # Denormalizing ground truth
        # y_true = t.multiply(y_true, self.maxValue)
        y_true = y_true * self.var + self.mean
        # Denormalizing prediction
        # y_pred = t.multiply(y_pred, self.maxValue)
        y_pred = y_pred * self.var + self.mean
        # Calculating mse per image
        mse_image = t.true_divide(self.mse(y_true, y_pred), self.batch_size)
        return mse_image

    def mse_denormalized_per_pixel(self, y_true, y_pred):
        # Calculating mse per image
        mse_image = self.mse_denormalized_per_image(y_true, y_pred)
        # Calculating mse per pixel
        ipdb.set_trace()
        mse_pixel = t.true_divide(mse_image, self.n_pixels)
        return mse_pixel


# Convert output to binary mask and calculate metrics
class thresholded_mask_metrics:
    def __init__(self, test_data=None, threshold=None, var=None, mean=None):
        self._binarized_mse = tf.keras.losses.MeanSquaredError()
        self._acc = tf.keras.metrics.Accuracy()
        self._precision = tf.keras.metrics.Precision()
        self._recall = tf.keras.metrics.Recall()
        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = np.mean(test_data)
        self._TP = tf.keras.metrics.TruePositives()
        self._TN = tf.keras.metrics.TrueNegatives()
        self._FP = tf.keras.metrics.FalsePositives()
        self._FN = tf.keras.metrics.FalseNegatives()
        self.var = var
        self.mean = mean

    def binarize_mask(self, values):
        # Initialize TF values
        # zero = tf.cast(tf.constant(0), tf.float32)
        # one = tf.cast(tf.constant(1), tf.float32)
        # limit = tf.cast(tf.constant(self._threshold), tf.float32)
        ones = t.ones(values.shape)
        zeros = t.zeros(values.shape)
        # Replacing values for 0s and 1s
        values = t.where(values.cpu() >= self._threshold, ones, zeros)
        # values = t.where(values < limit, 0, values)
        return values

    def binarized_mse(self, y_true, y_pred):
        # Denormalize
        y_true, y_pred = (
            y_true * self.var + self.mean,
            y_pred * self.var + self.mean,
        )
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._binarized_mse(y_true, y_pred)

    def acc(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._acc(y_true, y_pred)

    def precision(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._precision(y_true, y_pred)

    def recall(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        return self._recall(y_true, y_pred)

    def f1_score(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate metrics
        precision = self._precision(y_true, y_pred)
        recall = self._recall(y_true, y_pred)
        return 2 * precision * recall / (precision + recall)

    def CSI(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate TP, TN, FP, FN
        TP = self._TP(y_true, y_pred)
        TN = self._TN(y_true, y_pred)
        FP = self._FP(y_true, y_pred)
        FN = self._FN(y_true, y_pred)
        # Calculate metrics
        return TP / (TP + FN + FP)

    def FAR(self, y_true, y_pred):
        # Binarize mask
        y_true = self.binarize_mask(y_true)
        y_pred = self.binarize_mask(y_pred)
        # Calculate TP, TN, FP, FN
        TP = self._TP(y_true, y_pred)
        TN = self._TN(y_true, y_pred)
        FP = self._FP(y_true, y_pred)
        FN = self._FN(y_true, y_pred)
        # Calculate metrics
        return FP / (TP + FP)


def model_persistence(x):
    return x[:, :, -1]


def acc(y_true, y_pred):
    return np.sum((np.asarray(y_true) == np.asarray(y_pred))) / (
        y_true.shape[0] * y_true.shape[1]
    )


def precision(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    return TP / (TP + FP)


def recall(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    return TP / (TP + FN)


def extract_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path
"""
