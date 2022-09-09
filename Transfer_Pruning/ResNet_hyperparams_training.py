### ------------------------------- CONSTANT PARAMS -------------------------------
IMAGE_PIXELS = 32
IMAGE_CHANNEL = 3
OPTIMIZER_NAME = ['Nestrov_Momentum','Adam','Momentum','AdaBound']
BLOCK_TYPE = ['Normal', 'BottleNeck']
DATASET_NAME = ['CIFAR10', 'CIFAR100']

### ------------------------------- Training Params -------------------------------

# Network Structure and Dataset
Dataset = DATASET_NAME[0]
n_res_block = 9
block_type = BLOCK_TYPE[0]

# Epoch num and batch size
MAX_EPOCH = 200
Batch_Size = 128

# Optimizer
init_lr = 0.1 # The learning rate
lr_drop_rate = 0.1
lr_drop_epoch = 80
opt_momentum = 0.9
opt = OPTIMIZER_NAME[0]

# Regularizer
m_ridge = 8e-4
use_data_aug = True

bn_momentum = 0.99
bn_var_epsilon = 1e-5


# Loaded from other model
model_loaded = True 
model = './Model_ResNet56/ResNet56_2019-02-15_23:04:37_acc_94.11_new.npy'

