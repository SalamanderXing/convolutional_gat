import torch
import torch.nn as nn

MODEL_TYPE = "spatial"
PREPROCESSED_FOLDER = "/mnt/kmni_dataset/20_plus_preprocessed"
MAPPING_TYPE = "conv"
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
