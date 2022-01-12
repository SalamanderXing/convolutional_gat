import torch
import torch.nn as nn
from convolutional_gat.model import TemporalModel

MODEL = TemporalModel(time_steps=4, conv=True)
PREPROCESSED_FOLDER = "convolutional_gat/preprocessed"
# PREPROCESSED_FOLDER = "/mnt/kmni_dataset/new_preprocessed"
DATASET = "kmni"
EPOCHS = 15
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 1.0
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (80, 80)
