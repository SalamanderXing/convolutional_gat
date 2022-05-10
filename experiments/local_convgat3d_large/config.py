import torch
import torch.nn as nn

MODEL_TYPE = "convgat3d"
PREPROCESSED_FOLDER = "/mnt/kmni_dataset/20_latest"
MAPPING_TYPE = "conv"
DATASET = "kmni"
EPOCHS = 30
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 4
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 0.95
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (80, 80)
REDUCE_LR_ON_PLATEAU = True
N_HEADS_PER_LAYER = (1,)
