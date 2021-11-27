import torch
import torch.nn as nn
from convolutional_gat.model import TemporalModel
from convolutional_gat.data_loader import Task

MODEL = TemporalModel()
EPOCHS = 30
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
LEARNING_RATE = 0.001
TASK = Task.predict_next
LR_STEP = 1
GAMMA = 1.0
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (16, 16)
