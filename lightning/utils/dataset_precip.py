import torch as t
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import ipdb


class precipitation_maps_oversampled_h5(Dataset):
    def __init__(
        self,
        in_file,
        num_input_images,
        num_output_images,
        *,
        hparams,
        train=True,
        transform=None,
        lag=6,
    ):
        super().__init__()
        self.hparams = hparams
        self.lag = lag
        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, "r")[
            "train" if train else "test"
        ]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None
        self.coordinates = (  # coordinates of areas of interest within the larger image
            (201, 38),
            (201 - 80, 81),
            (201 - 80 + 4, 81 + 92),
            (205, 140),
            (29, 190),
            (29 + 10, 186 - 85),
        )

    def make_graph_from_image_sequence(self, data):
        size = self.hparams.subsample_size
        crops = []
        for i, (x, y) in enumerate(self.coordinates):
            x0 = x
            x1 = x + size
            if x1 > data.shape[1]:
                x1 = data.shape[1]
                x0 = x1 - size
            y0 = y
            y1 = y + size
            if y1 > data.shape[2]:
                y1 = data.shape[2]
                y0 = y1 - size
            crop = data[:, x0:x1, y0:y1]
            crops.append(crop)
        return np.stack(crops)

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024 ** 3)[
                "train" if self.train else "test"
            ]["images"]
        imgs = np.array(self.dataset[index], dtype="float32")
        # print("******SHAPE: ", imgs.shape)
        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        # target_img = imgs[-1] # Modified
        target_img = imgs[self.num_input :]
        input_img = t.from_numpy(
            self.make_graph_from_image_sequence(input_img)
        ).permute(2, 3, 1, 0)
        target_img = t.from_numpy(
            self.make_graph_from_image_sequence(target_img)
        ).permute(2, 3, 1, 0)
        target_img = target_img[
            : self.hparams.subsample_size,
            : self.hparams.subsample_size,
            :,
            : self.hparams.nregions,
        ]
        input_img = input_img[
            : self.hparams.subsample_size,
            : self.hparams.subsample_size,
            :,
            : self.hparams.nregions,
        ]
        if self.hparams.model == "UNetDS_Attention":
            # in this case batch size should be 1
            target_img = target_img.squeeze()
            input_img = input_img.squeeze()
            target_img = target_img.permute(3, 2, 0, 1)
            input_img = input_img.permute(3, 2, 0, 1)
        """
        if self.hparams.model != "GAT3D":
            input_img = input_img[:, :, :, 0].permute(2, 0, 1)
            target_img = target_img[:, :, :, 0].permute(2, 0, 1)
        """
        return input_img, target_img

    def __len__(self):
        return self.samples - 1


# define a Bunch class
class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_oversampled_dataset(downsample_size=80):
    import json

    hparams = Bunch(
        json.loads(
            """{
        "model": "GAT3D",
        "n_channels": 4,
        "n_classes": 4,
        "kernels_per_layer": 2,
        "bilinear": true,
        "reduction_ratio": 16,
        "lr_patience": 4,
        "num_input_images": 9,
        "num_output_images": 9,
        "valid_size": 0.1,
        "use_oversampled_dataset": true,
        "logger": true,
        "checkpoint_callback": true,
        "early_stop_callback": false,
        "default_root_dir": null,
        "gradient_clip_val": 0,
        "process_position": 0,
        "num_nodes": 1,
        "num_processes": 1,
        "gpus": 1,
        "auto_select_gpus": false,
        "num_tpu_cores": null,
        "log_gpu_memory": null,
        "progress_bar_refresh_rate": 1,
        "overfit_pct": 0.0,
        "track_grad_norm": -1,
        "check_val_every_n_epoch": 1,
        "fast_dev_run": false,
        "accumulate_grad_batches": 1,
        "max_epochs": 1000,
        "min_epochs": 1,
        "max_steps": null,
        "min_steps": null,
        "train_percent_check": 1.0,
        "val_percent_check": 1.0,
        "test_percent_check": 1.0,
        "val_check_interval": 1.0,
        "log_save_interval": 100,
        "row_log_interval": 10,
        "distributed_backend": null,
        "precision": 32,
        "print_nan_grads": false,
        "weights_summary": "full",
        "weights_save_path": null,
        "num_sanity_val_steps": 2,
        "truncated_bptt_steps": null,
        "resume_from_checkpoint": null,
        "profiler": null,
        "benchmark": false,
        "deterministic": false,
        "reload_dataloaders_every_epoch": false,
        "auto_lr_find": false,
        "replace_sampler_ddp": true,
        "progress_bar_callback": true,
        "terminate_on_nan": false,
        "auto_scale_batch_size": false,
        "amp_level": "O1",
        "dataset_folder": "train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5",
        "experiment_save_path": "/home/bluesk/Documents/convolutional_gat/lightning/../experiments/local_temporal_conv_overlapping_v_large",
        "batch_size": 6,
        "learning_rate": 0.001,
        "epochs": 200,
        "overlapping_nodes": true,
        "size": "large",
        "subsample_size": 80,
        "nregions": 6,
        "es_patience": 30,
        "save_path": "/home/bluesk/Documents/convolutional_gat/lightning/../experiments/local_temporal_conv_overlapping_v_large"
    }"""
        )
    )
    hparams.subsample_size = downsample_size

    return DataLoader(
        precipitation_maps_oversampled_h5(
            in_file=hparams.dataset_folder,
            num_input_images=hparams.num_input_images,
            num_output_images=hparams.num_output_images,
            train=False,
            transform=None,
            hparams=hparams,
        ),
        batch_size=50,
        shuffle=True,
    )


if __name__ == "__main__":
    dataloader = get_oversampled_dataset()
    for a, b in dataloader:
        print(a.shape)
"""
class precipitation_maps_h5(Dataset):
    def __init__(
        self,
        in_file,
        num_input_images,
        num_output_images,
        train=True,
        transform=None,
        hparams=None,
    ):
        super(precipitation_maps_h5, self).__init__()
        self.hparams = hparams
        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, "r")[
            "train" if train else "test"
        ]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.sequence_length = num_input_images + num_output_images

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images - (
            num_input_images + num_output_images
        )
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_name, "r", rdcc_nbytes=1024 ** 3
            )["train" if self.train else "test"]["images"]
        imgs = np.array(
            self.dataset[index : index + self.sequence_length], dtype="float32"
        )

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return self.size_dataset




class precipitation_maps_classification_h5(Dataset):
    def __init__(
        self,
        in_file,
        num_input_images,
        img_to_predict,
        train=True,
        transform=None,
    ):
        super(precipitation_maps_classification_h5, self).__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, "r")[
            "train" if train else "test"
        ]["images"].shape

        self.num_input = num_input_images
        self.img_to_predict = img_to_predict
        self.sequence_length = num_input_images + img_to_predict
        self.bins = np.array([0.0, 0.5, 1, 2, 5, 10, 30])

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images - (num_input_images + img_to_predict)
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_name, "r", rdcc_nbytes=1024 ** 3
            )["train" if self.train else "test"]["images"]
        imgs = np.array(
            self.dataset[index : index + self.sequence_length], dtype="float32"
        )

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        # put the img in buckets
        target_img = imgs[-1]
        # target_img is normalized by dividing through the highest value of the training set. We reverse this.
        # Then target_img is in mm/5min. The bins have the unit mm/hour. Therefore we multiply the img by 12
        buckets = np.digitize(target_img * 47.83 * 12, self.bins, right=True)

        return input_img, buckets

    def __len__(self):
        return self.size_dataset
"""
