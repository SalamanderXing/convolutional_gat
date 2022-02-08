import torch
import torch.nn as nn

MODEL_TYPE = "spatial"
PREPROCESSED_FOLDER = "/mnt2/20_plus_preprocessed"
MAPPING_TYPE = "linear"
DATASET = "kmni"
EPOCHS = 20
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 0.95
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (20, 20)
REDUCE_LR_ON_PLATEAU = True
