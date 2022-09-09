import numpy as np

# ------------------------------- Train Params -------------------------------
model_loaded = True
model_name = './LeNet5_Model/LeNet-5_Ind_0_2019-12-11_11:35:31_acc_99.26_FLOP_100.0_Param_100.0.npy'

batch_size = 200
MAX_EPOCH = 100
opt = 'Adam'
learning_rate = 5e-4 
m_ridge =  7e-5

# ------------------------------- Prune Params -------------------------------
Prune_Step = 3

conv1_sel_list = [ 10, 5, 0]
conv1_rmv_list = [ 10, 5, 0]
conv2_sel_list = [ 30, 8, 1]
conv2_rmv_list = [ 30, 8, 1]
fla_sel_list   = [ 0,  0, 0]
fla_rmv_list   = [ 0,  0, 0]
a1_sel_list    = [ 400, 55, 5]
a1_rmv_list    = [ 400, 55, 5]




