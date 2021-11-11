#!/usr/bin/env python
# coding: utf-8

# # ACML Homework: Convolutional Autoencoders
#

# This assigment shows how to use a deep convolutional autoencoder in order to reconstruct the images from CIFAR10 dataset. The goal of image reconstruction is to create a new set of images similar to the original input images.
#
# ### Running this notebook:
# Make sure you have the `torch, torchvision, numpy, matplotlib, tqdm, opencv` packages installed.

# In[1]:


import torch as t
import torchvision  # package contains the image data sets that are ready for use
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (20, 10)
print(t.__version__)


# The first step is to pre-process the data.
# The names of the classes over which the dataset is distributed will be defined. There are ten different classes of 32x32 color images. After we have decided on a class name, to make our model train faster, we need to normalize the image.
# In this section we:
# - downloaded the dataset CIFAR10
# - defined the image classes, loaders
# - prepared data loaders
#
# Notice that by downloading the data via PyTorch, we already get it normalized in the range `[0,1]`, which is perfect for us. For this reason, the last layer of all the moels contained in this notebook will have a sigmoid activation function, because it naturally outputs values in that range.

# In[2]:


# batch_size = the number of samples that will be fed into the model at the same time, after which the loss value will be computed
def get_loaders(train_batch_size=64, test_batch_size=100):
    transform = transforms.Compose([transforms.ToTensor()])
    # Download the training and test datasets
    train_val_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                          shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    # Define the image classes
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    val_size = 0.1
    num_workers = 2
    print(len(test_set) / (len(train_val_set) + len(test_set)))
    # obtain training indices that will be used for validation
    # validation set is 10% of the training set
    num_train_val = len(train_val_set)
    indices = t.randperm(num_train_val)
    split = int(np.floor(val_size * num_train_val))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = t.utils.data.DataLoader(
        train_val_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = t.utils.data.DataLoader(
        train_val_set,
        batch_size=test_batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )
    test_loader = t.utils.data.DataLoader(
        train_val_set, batch_size=test_batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# In[3]:


# functions to show an image


def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_imgs(imgs):
    imshow(torchvision.utils.make_grid(imgs))


# get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# show images
# print labels #print(" ".join("%5s" % classes[labels[j]] for j in range(train_batch_size)))


# Convolutional Autoencoder is used in image reconstruction to learn the optimal filters in order to reduce reconstruction errors.
# In both arhitectures we start by implementing the encoder. It is effectively a deep convolutional network in which we use strided convolutions to scale down the image layer by layer. After we have a decoder which is the flipped version of the encoder.

# ### Model 1 structure
# The first model to be tested will be using maxUnpooling as a form of upsampling. As the name suggests, this is the partial inverse of the maxPooling. Partial because all the non-maximal value are lost during the poolling operation. MaxPool will set all the non-maximal values to zero.

# In[11]:


import torch.nn as nn
import torch.nn.functional as F


class ConvAEUnpool(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(3, 8, 3, padding="same")
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(8, 12, 3, padding="same")
        self.conv3 = nn.Conv2d(12, 16, 3, padding="same")
        # decoder
        self.unpool = nn.MaxUnpool2d(2)
        self.dec_conv1 = nn.Conv2d(16, 12, 3, padding="same")
        self.dec_conv2 = nn.Conv2d(12, 8, 3, padding="same")
        self.dec_conv3 = nn.Conv2d(8, 3, 3, padding="same")

    def forward(self, x):
        # encoder
        x = F.relu(self.conv1(x))
        x, pool_idx_1 = self.pool(x)
        x = F.relu(self.conv2(x))
        x, pool_idx_2 = self.pool(x)
        x = F.relu(self.conv3(x))
        x, pool_idx_3 = self.pool(x)
        # decoder

        print(f"{t.prod(t.tensor(x.shape[1:]))=}")
        x = self.unpool(x, pool_idx_3)
        x = F.relu(self.dec_conv1(x))
        x = self.unpool(x, pool_idx_2)
        x = F.relu(self.dec_conv2(x))
        x = self.unpool(x, pool_idx_1)
        x = t.sigmoid(self.dec_conv3(x))
        return x


# ### Model 2 structure
# The second model to be tested will be using transpose convolution in the decoder. By puttig `stride=2` we can effectively double the size of the output, i.e., upsample.

# In[ ]:


class ConvAETransposeConv(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.conv1 = nn.Conv2d(3, 8, 3, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 12, 3, padding="same")
        self.conv3 = nn.Conv2d(12, 16, 3, padding="same")
        # decoder
        self.dec_conv1 = nn.ConvTranspose2d(16, 12, 2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(12, 8, 2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(8, 3, 2, stride=2)

    def forward(self, x):
        # encoder
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # decoder

        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = t.sigmoid(self.dec_conv3(x))
        return x


# ### Training and Testing
# In PyTorch, unlike Keras, one has to create the training loop and the logging from schratch. We also have to explicitly deactivate the autograd engine during testing, zeroing the gradients and tell pytorch to run the parameters optimization.
#
# After a few attempts we have found that a batch size of 32 is a good fit. A smaller batch size increase the likelihood to find a global (not local) minimum, but it also slows down training and it might make it more unstable as well. The test batch size instead should be as large as our GPU can afford.
# We have also experimented with step learning rate scheduler, which decreases the learning rate during training. However this did not increase performances. We also tried out BCELoss but this as well did not improve significantly the results.

# In[31]:


from tqdm import tqdm


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
    model_class=ConvAEUnpool,
    train_batch_size=64,
    test_batch_size=200,
    epochs=10,
    lr=0.001,
    lr_step=1,
    gamma=1.0,  # 1.0 means disabled
    plot=True,
):
    train_loader, val_loader, test_loader = get_loaders(
        train_batch_size, test_batch_size
    )
    print(f"Len train set: {len(train_loader)}")
    device = t.device(
        "cuda" if t.cuda.is_available() else "cpu"
    )  # Select the GPU device, if there is one available.
    # device = t.device('cpu')
    print(device)
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
        plot_reconstruction(model, device, test_loader)
    return history, test_loss


# In[114]:


def plot_history(history, title="Training history"):
    print(history)
    plt.plot(t.arange(len(history)), [h[0] for h in history], label="Train loss")
    plt.plot(t.arange(len(history)), [h[1] for h in history], label="Val loss")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_reconstruction(model, device, loader):
    with t.no_grad():
        model.eval()
        for batch, _ in loader:
            batch = batch[:8]
            recon_batch = model(batch.to(device))
            plot_imgs(batch)
            plot_imgs(recon_batch.cpu())
            break


# ### Training Model 1 for 10 epochs
#

# In[33]:


history = train(ConvAEUnpool, 32, 600, 10, 0.001, 1, 1.0)


# ### Training Model 2 for 10 epochs

# In[34]:


history = train(ConvAETransposeConv, 32, 600, 10, 0.001, 1, 1.0)


# ### Conclusion
# Clearly MaxUnpooling performs much better during a 10-epochs training.

# ### Training a new Model
#
# If we are allowed to increase the size of the latent space, we can easlily create a model that performs much better. Here train another model with a latent space dimension of 48x4x4=768. We use Transpose convolution as an upsampling technique. The training remains exacly the same, but we use a batch size of 16.

# In[37]:


from torch import nn


class AE2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(t.prod(t.tensor(encoded.shape[1:])))
        x = self.decoder(x)
        return x


# In[38]:


history = train(AE2, 16, 400, 20, 0.001, 1, 1.0)


# ### Results
# This model, trained fro 20 epochs clearly outperforms the two we have previously developed. However we need to take into accound both the longer training time and especially the compression rate.
# The initial image size is 3x32x32=3072. Model 1 and Model 2 have both a latent space of size 256, which means a compression rate of 12.
# This new model on the other hand has a compression rate of only 4, its latent space of size 768.

# ## Colorization
#
# First we need to create a function which, as suggested in the assignment, separates the chrominance. For this we can use the YUV format and the opencv library to convert our RGB images. The `get_yuv` function does just that.

# In[12]:


import cv2
import numpy as np

# this function was only used to plot the grayscale image
def get_yuv_images(img):
    img = (img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(img_yuv)
    lut_u = np.array([[[i, 255 - i, 0] for i in range(256)]], dtype=np.uint8)
    lut_v = np.array([[[0, 255 - i, i] for i in range(256)]], dtype=np.uint8)
    # Convert back to BGR so we can apply the LUT and stack the images
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    u_mapped = cv2.LUT(u, lut_u)
    v_mapped = cv2.LUT(v, lut_v)

    return y, u_mapped, v_mapped


def get_yuv(img):
    img = (img.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    img_yuv = t.from_numpy(cv2.cvtColor(img, cv2.COLOR_RGB2YUV)) / 255
    return img_yuv[:, :, :1].transpose(0, 2), img_yuv[:, :, 1:].transpose(0, 2)


# We also create some dataloaders for this new dataset

# In[13]:


import torch as t


def get_loaders_colorization(train_batch_size=64, test_batch_size=100):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            get_yuv,  # We add the transformation to YUV format to the preprocessing
        ]
    )
    # Download the training and test datasets
    train_val_set = torchvision.datasets.CIFAR10(
        root="./colorization_data", train=True, download=True, transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./colorization_data", train=False, download=True, transform=transform
    )
    val_size = 0.1
    num_workers = 2
    print(len(test_set) / (len(train_val_set) + len(test_set)))
    # obtain training indices that will be used for validation
    # validation set is 10% of the training set
    num_train_val = len(train_val_set)
    indices = t.randperm(num_train_val)
    split = int(np.floor(val_size * num_train_val))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = t.utils.data.DataLoader(
        train_val_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = t.utils.data.DataLoader(
        train_val_set,
        batch_size=test_batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )
    test_loader = t.utils.data.DataLoader(
        train_val_set, batch_size=test_batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# ### The Colorization Model
#
# Here we have simply adapted the model used previously, by changing the input channels to 1 and the output channels to 2 we can have our model predict the "UV" components of an image given the "Y" component.

# In[15]:


from torch import nn


class ConvColorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 2, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print(t.prod(t.tensor(encoded.shape[1:])))
        decoded = self.decoder(encoded)
        return decoded


# In[87]:


def imshow(img):
    plt.imshow(img.T)
    plt.show()


def plot_imgs(imgs_colors, imgs_structures):
    imgs = t.cat((imgs_colors, imgs_structures), dim=1)
    imgs = (imgs * 255).numpy().astype(np.uint8)

    rgb_imgs = t.stack(
        [t.from_numpy(cv2.cvtColor(img.T, cv2.COLOR_YUV2RGB)).T for img in imgs]
    )
    # imshow(rgb_imgs[0])
    imshow(torchvision.utils.make_grid(rgb_imgs, nrow=1))


def plot_colorer(model, device, loader):
    with t.no_grad():
        model.eval()
        for (structure, colors), _ in loader:
            structure = structure[:8]
            colors = colors[:8]
            output = model(structure.to(device))
            plot_imgs(structure.cpu(), colors.cpu())
            plot_imgs(structure.cpu(), output.cpu())
            break


# ### Training and Testing
# Training and testing remains almost unchanged: we only need to take into account that now we don't have to predict the image itself but only two of its components.

# In[119]:


from tqdm import tqdm


def test_colorer(model, device, criterion, val_test_set):
    model.eval()
    with t.no_grad():
        running_loss = 0.0
        total_length = 0
        for data in tqdm(val_test_set):
            (structures, colors), _ = data

            inputs = structures.to(device)
            outputs = model(inputs)
            colors = colors.to(device)
            running_loss += t.sum((colors - outputs) ** 2).item()
            total_length += len(inputs)
    model.train()
    return running_loss / total_length


def train_colorer(
    model_class=ConvColorer,
    train_batch_size=64,
    test_batch_size=200,
    epochs=10,
    lr=0.001,
    lr_step=1,
    gamma=1.0,
    plot=True,
):
    train_loader, val_loader, test_loader = get_loaders_colorization(
        train_batch_size, test_batch_size
    )
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(device)
    model = model_class().to(device)
    # optimizer = the procedure for updating the weights of our neural network
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=gamma)
    history = []
    for epoch in range(epochs):

        model.train()
        print(f"\nEpoch: {epoch + 1}")
        running_loss = 0.0
        total_length = 0
        for param_group in optimizer.param_groups:
            print(f"LR: {param_group['lr']}")
        for data in tqdm(train_loader):
            (structures, colors), _ = data
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            inputs = structures.to(device)
            colors = colors.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, colors)
            loss.backward()
            optimizer.step()
            total_length += len(inputs)
            running_loss += t.sum((colors - outputs) ** 2).item()
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        print(f"Train loss: {round(train_loss, 6)}")
        val_loss = test_colorer(model, device, criterion, val_loader)
        print(f"Val loss: {round(val_loss, 6)}")
        history.append((train_loss, val_loss))
    test_loss = test_colorer(model, device, criterion, test_loader)
    print(f"Test loss: {round(test_loss, 6)}")

    train_history = np.array([h[0] for h in history])
    max_th = np.max(train_history)
    min_th = np.min(train_history)
    train_history = (train_history - min_th) / (max_th - min_th)

    val_history = np.array([h[1] for h in history])
    max_vh = np.max(val_history)
    min_vh = np.min(val_history)
    val_history = (val_history - min_vh) / (max_vh - min_vh)
    history = list(zip(train_history, val_history))
    if plot:
        plot_history(history, "Normalized training history")
        plot_colorer(model, device, test_loader)
    return history, test_loss


# In[125]:


history, test_loss = train_colorer(
    model_class=ConvColorer,
    train_batch_size=32,
    test_batch_size=200,
    epochs=13,
    lr=0.001,
    lr_step=1,
    gamma=0.90,
    plot=True,
)


# ## Results
# The results aren't perfect but it's clear that our model is indeed learning to color images. There are a number of things that could be inproved, among them:
# - **tune further the training hyperparameters**
# - we could use a deeper neural netowrk
# - training for more epochs
# - experiment more with hyperparameters
# - Use batch normalization
# - Add residual connections in case of a much deeper neural network
# - Try more image encoding schemes

# ## Conclusions
# A variety of models were tested: Autoencoders are a powerful tool for any task that involves unsupervised training, among which (partial) image reconstruction (like coloring) or denoising.
