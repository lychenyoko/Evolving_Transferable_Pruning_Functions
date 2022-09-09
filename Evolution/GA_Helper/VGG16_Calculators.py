import numpy as np

# ---------------------------- FLOP and Param Calculator ----------------------------
VGG16_MAP_SIZE = [32,32,16,16,8,8,8,4,4,4,2,2,2,1]
VGG16_KER_SIZE = [3,3,3,3,3,3,3,3,3,3,3,3,3,1]

def GetFeamapNum(VGG16_model):
    feamap_num = [int(bias.shape[0]) for bias in VGG16_model.network_param_list[1:-2:2]]
    return feamap_num

ori_FLOP = 313463808
def VGG16_FLOPCal(VGG16_model):
    feamap_num = GetFeamapNum(VGG16_model)
    
    total_FLOP = 0
    for ind,map_size in enumerate(VGG16_MAP_SIZE):
        tmp = np.power(map_size * VGG16_KER_SIZE[ind], 2) * feamap_num[ind]
        if ind == 0:
            total_FLOP = tmp * 3 + total_FLOP
        else:
            total_FLOP = tmp * feamap_num[ind - 1] + total_FLOP
    total_FLOP = total_FLOP + feamap_num[-1] * 10 # The MLP FLOPs for CIFAR10
    return total_FLOP/ori_FLOP

ori_Param = 14977728
def VGG16_ParamCal(VGG16_model):
    feamap_num = GetFeamapNum(VGG16_model)
    
    total_param = 0
    for ind,map_size in enumerate(VGG16_MAP_SIZE):
        tmp = np.power(VGG16_KER_SIZE[ind], 2) * feamap_num[ind]
        if ind == 0:
            total_param = tmp * 3 + total_param
        else:
            total_param = tmp * feamap_num[ind - 1] + total_param
    total_param = total_param + feamap_num[-1] * 10 # The MLP params for CIFAR10
    return total_param/ori_Param
