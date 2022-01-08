import torch
import torch.nn as nn
from convolutional_gat.model import TemporalModel

MODEL = TemporalModel()
PREPROCESSED_FOLDER = "/mnt/kmni_dataset/preprocessed"
DATASET = "kmni"
EPOCHS = 5
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 16
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 1.0
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (35, 35)
# DOWNSAMPLE_SIZE = (1, 16)
