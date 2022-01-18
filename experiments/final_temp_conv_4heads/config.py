import torch
import torch.nn as nn
from convolutional_gat.model import TemporalModel4h

MODEL = TemporalModel4h
PREPROCESSED_FOLDER = "convolutional_gat/preprocessed"
MAPPING_TYPE = "conv"
DATASET = "kmni"
EPOCHS = 20
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 0.1
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (80, 80)
