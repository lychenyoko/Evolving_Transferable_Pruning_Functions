import numpy as np
### ------------------------------- CONSTANT PARAMS -------------------------------
NUM_CLASS = 10
IMAGE_PIXELS = 32
IMAGE_CHANNEL = 3
OPTIMIZER_NAME = ['Nestrov_Momentum','Adam']

### ------------------------------- Training Params -------------------------------
MAX_EPOCH = 100
Batch_Size = 128

# Optimizer
MOMENTUM = 0.9
opt = OPTIMIZER_NAME[0]
init_lr = 6e-3 # The learning rate
lr_drop_rate = 0.28 # the rate that lr drop
lr_drop_epoch1 = 40 # the # of epochs where lr will drop
lr_drop_epoch2 = 80

# Regularizer
m_ridge = 0.001
dropout_keep_prob = 0.85
use_data_aug = True

# Initialization
model_loaded = True
m_model = './VGG16_Model/VGG16_2018-12-02_14:50:05_acc_93.92.npy'

### ------------------------------- Pruning Params -------------------------------
Prune_Step = 2
sel_list = [12,  12,  25,  25,  51,  51,  51, 102, 102, 102, 102, 102, 102, 102]
rmve_list = sel_list
