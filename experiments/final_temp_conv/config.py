import torch
import torch.nn as nn

MODEL_TYPE = "temporal"
PREPROCESSED_FOLDER = "convolutional_gat/preprocessed"
MAPPING_TYPE = "conv"
DATASET = "kmni"
EPOCHS = 20
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LEARNING_RATE = 0.0001
LR_STEP = 1
GAMMA = 0.1
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (80, 80)
REDUCE_LR_ON_PLATEAU = True