import torch
import torch.nn as nn

MODEL_TYPE = "unet"
PREPROCESSED_FOLDER = "/mnt/kmni_dataset/20_full_size"
MAPPING_TYPE = "linear"
DATASET = "kmni"
EPOCHS = 30
TRAIN_BATCH_SIZE = 3
TEST_BATCH_SIZE = 5
LEARNING_RATE = 0.001
LR_STEP = 1
GAMMA = 0.95
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam
DOWNSAMPLE_SIZE = (None, 20)
