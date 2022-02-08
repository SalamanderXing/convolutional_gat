import torch
import torch.nn as nn
from convolutional_gat.baseline_model import BaselineModel2D

MODEL = BaselineModel2D
PREPROCESSED_FOLDER = "/mnt/kmni_dataset/20_preprocessed"
MAPPING_TYPE = "linear"
DATASET = "kmni"
EPOCHS = 20
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 0.95
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (20, 20)
