import torch
import torch.nn as nn
from convolutional_gat.model import TemporalModel2l, SpatialModel

MODEL = TemporalModel2l
PREPROCESSED_FOLDER = "convolutional_gat/preprocessed"
MAPPING_TYPE = "smaat_unet"
DATASET = "kmni"
EPOCHS = 20
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 0.1
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (80, 80)
