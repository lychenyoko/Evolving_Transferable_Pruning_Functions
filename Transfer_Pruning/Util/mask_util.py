import numpy as np

Three_Fea_Map_Size = 3 
Two_Mask_Per_Block = 2


# Create The Default Mask with all True for the Network
def GetDefaultMaskList(net_mask_pl):
    net_mask = []
    for blk_list_i in range(len(net_mask_pl)):
        stage_mask = []
        for blk_j in range(len(net_mask_pl[blk_list_i])):
            block_mask = []
            for lay_k in range(len(net_mask_pl[blk_list_i][blk_j])):
                default_layer_mask = np.array([True] * net_mask_pl[blk_list_i][blk_j][lay_k].shape[0])
                block_mask.append(default_layer_mask)
            stage_mask.append(block_mask)
        net_mask.append(stage_mask)
    return net_mask


# Get the Mask Dictionary for tf session running
def GetMaskDict(net_mask_pl,mask_list):
    mask_dict = {}
    for stage_i in range(len(net_mask_pl)):
        for blk_j in range(len(net_mask_pl[stage_i])):
            for lay_k in range(len(net_mask_pl[stage_i][blk_j])):
                pl_key = net_mask_pl[stage_i][blk_j][lay_k]
                mask_val = mask_list[stage_i][blk_j][lay_k]
                mask_dict[pl_key] = mask_val
    return mask_dict


# Helper Function For Printing Mask Shape
def Print_Mask_Shape(net_mask_list):
    net_shape = []
    for stage_i in range(len(net_mask_list)):
        stage_shape = []
        for blk_j in range(len(net_mask_list[stage_i])):
            block_shape = []
            for lay_k in range(len(net_mask_list[stage_i][blk_j])):
                layer_shape = sum(net_mask_list[stage_i][blk_j][lay_k])
                block_shape.append(layer_shape)
            stage_shape.append(block_shape)
        net_shape.append(stage_shape)

    for stage in net_shape:
        print(stage)
