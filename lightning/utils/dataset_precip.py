import torch as t
from torch.utils.data import Dataset
import h5py
import numpy as np
import ipdb


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
        self.size_dataset = self.n_images - (num_input_images + num_output_images)
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
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024 ** 3)[
                "train" if self.train else "test"
            ]["images"]
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
        lag=4,
    ):
        super(precipitation_maps_oversampled_h5, self).__init__()
        self.hparams = hparams
        self.lag = 4
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
        width, height = 80, 80
        crops = []
        for x, y in self.coordinates:
            crop = data[:, x : x + width, y : y + width]
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
            self.make_graph_from_image_sequence(input_img)[:, :4]
        ).permute(2, 3, 1, 0)
        target_img = t.from_numpy(
            self.make_graph_from_image_sequence(target_img)[:, :4]
        ).permute(2, 3, 1, 0)
        target_img = target_img[
            : self.hparams.subsample_size, : self.hparams.subsample_size
        ]
        input_img = input_img[
            : self.hparams.subsample_size, : self.hparams.subsample_size
        ]
        """
        if self.hparams.model != "GAT3D":
            input_img = input_img[:, :, :, 0].permute(2, 0, 1)
            target_img = target_img[:, :, :, 0].permute(2, 0, 1)
        """
        # print(f"{input_img.shape} {target_img.shape}")

        return input_img, target_img

    def __len__(self):
        return self.samples


class precipitation_maps_classification_h5(Dataset):
    def __init__(
        self, in_file, num_input_images, img_to_predict, train=True, transform=None,
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
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024 ** 3)[
                "train" if self.train else "test"
            ]["images"]
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
