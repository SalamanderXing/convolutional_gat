import torch
from ...GAT3D.GATLayerTemporal import TemporalModel
from ...data_loader import Task

MODEL = TemporalModel
EPOCHS = 10
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
LEARNING_RATE = 0.001
TASK = Task.predict_next
LR_STEP = 1
GAMMA = 1.0
PLOT = False
CRITERION = nn.MSELoss()
OPTIMIZER = TORCH.optim.Adam