import torch
import torch.nn as nn
from convolutional_gat.baseline_model import BaselineModel

MODEL = BaselineModel
PREPROCESSED_FOLDER = "/mnt/kmni_dataset/20_preprocessed"
MAPPING_TYPE = "linear"
DATASET = "kmni"
EPOCHS = 10
TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 10
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 0.5
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (20, 20)
