### ------------------------------- CONSTANT PARAMS -------------------------------
IMAGE_PIXELS = 32
IMAGE_CHANNEL = 3
OPTIMIZER_NAME = ['Nestrov_Momentum','Adam','Momentum']
BLOCK_TYPE = ['Normal', 'BottleNeck']
DISCRIMINANT_FUNC_LIST = ['DI','AbsSNR', 'SymDiv', 'TwoT', 'FDR', 'Random']
DATASET_NAME = ['CIFAR10', 'CIFAR100']
PRUNING_MODE = ['Uniform', 'Predefined','Automatic_Ratio','Automatic_FLOPs']
INTERMEDIATE_DISTILLATION_MODE = ['Hint', 'DCA']

ResNet164_Full_Shape = [
    [[64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16], [64, 16, 16]],
    [[64, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32], [128, 32, 32]],
    [[128, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64], [256, 64, 64]]
]

ResNet_56_Full_Shape = [
    [[16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
    [[16, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]],
    [[32, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]
]

ResNet110_Full_Shape = [
[[16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
[[16, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]],
[[32, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64]],
]
### ------------------------------- Training Params -------------------------------
# Network Structure and Dataset
Dataset = DATASET_NAME[1]
n_res_block = 9
block_type = BLOCK_TYPE[0]

# Epoch num and batch size
MAX_EPOCH = 200
Batch_Size = 128

# Optimizer
init_lr = 0.05 # The learning rate
lr_drop_rate = 0.13
lr_drop_epoch = 80
opt_momentum = 0.9
opt = OPTIMIZER_NAME[0]

# Regularizer
m_ridge = 1e-3
use_data_aug = True

bn_momentum = 0.99
bn_var_epsilon = 1e-5

# Loaded from other model
model_loaded = True
model = './Model_ResNet56/ResNet56_2020-04-21_04:46:39_acc_73.27_FLOP_100.0_Param_100.67/ResNet56_2020-04-21_04:46:39_acc_73.27_FLOP_100.0_Param_100.67.npy'

# Knowledge Distillation
teacher_model = './Model_ResNet56/ResNet56_2020-04-21_04:46:39_acc_73.27_FLOP_100.0_Param_100.67/ResNet56_2020-04-21_04:46:39_acc_73.27_FLOP_100.0_Param_100.67.npy'
output_kd_lambda = 1.0
inter_kd_mode = 'DCA'
inter_kd_lambda = 0
inter_kd_block = 'Res16_2'
inter_kd_layer = 'Relu2 Output'
dca_mode = 'coarse'
dca_learn_freq = 100 # Learn the DCA weight every 5 epochs


### ------------------------------- Pruning Params -------------------------------
need_pruning = True
Pfunc = DISCRIMINANT_FUNC_LIST[2]
pruning_mode = PRUNING_MODE[0]
prune_ratio = 0.45
num_rmve_lay = None
prune_step = 1
prune_multiple = None

full_network_shape = ResNet_56_Full_Shape
pre_rmve_list = [
    [[6, 5], [0, 4], [11, 10], [8, 8], [8, 10], [15, 15], [6, 1], [7, 0], [7, 11]],
    [[4, 8], [22, 28], [8, 24], [30, 30], [18, 30], [30, 30], [8, 4], [10, 12], [12, 14]],
    [[20, 52], [16, 0], [18, 18], [32, 44], [48, 52], [32, 32], [18, 18], [0, 8], [12, 20]]
]
pre_sel_list = pre_rmve_list

# Hierarchical Label Loading
cifar_ytr_coarse_file = './ResNet_Multi_Resolution/CIFAR_100_Coarse_Label/cifar100_coarse_ytr_confusion_matrix_spectral_clustering.npy'
block_threshold = 14
front_label = 'coarse'
rear_label = 'fine'


