import numpy as np


def Get_BNScaling_Score(mResNet, sess):
    BN_obj_list = mResNet.bn_list[:-1] # the final fc BN is not useful
    BN_scale_list = sess.run([BN.scale for BN in BN_obj_list])
    return BN_scale_list


### Get the convolutional kernel, Current method just get the convolutional kernel without merging BN layer into it
def Get_Conv_Ker_Numpy(mResNet, sess):
    Conv_Ker_Tensor_List = [mResNet.W_conv0]
    for block_list in mResNet.block_list:
        for blk in block_list:
            Conv_Ker_Tensor_List += blk.weight_list
    
    Conv_Ker_np_List = sess.run(Conv_Ker_Tensor_List[:-1])        
    return Conv_Ker_np_List


def Get_Kernel_L1_Norm_List(mKer):
    '''
    Args:
        mKer: a 4D Kernel [w, h, inp, out]
    '''
    
    Layer_L1_Norm = []
    for out_ind in range(mKer.shape[-1]):
        sel_ker = mKer[..., out_ind]
        flattened_sel_ker = sel_ker.reshape(-1)
        L1_Norm = np.linalg.norm(flattened_sel_ker, ord=1)
        Layer_L1_Norm.append(L1_Norm)
    return Layer_L1_Norm


def Get_Kernel_GM_Score_List(mKer):
    '''
    Args:
        mKer: a 4D Kernel [w, h, inp, out]
    '''
    
    Layer_GM_Score = []
    for out_ind in range(mKer.shape[-1]):
        channel_gm_score = Get_Channel_GM_Score(mKer = mKer, index = out_ind)
        Layer_GM_Score.append(channel_gm_score)
    return Layer_GM_Score



def Get_Channel_GM_Score(mKer, index):
    '''
    Args:
        mKer: a 4D Kernel [w, h, inp, out]
        index: the index of the last dimension of mKer for GM Score calculation
    '''
    
    mask = [True] * mKer.shape[-1]
    mask[index] = False
    
    sel_ker = mKer[..., index]
    other_ker = mKer[..., mask]
    
    norm_score = 0
    for i in range(other_ker.shape[-1]):
        diff = sel_ker - other_ker[..., i]
        tmp = np.linalg.norm(diff.reshape(-1))
        norm_score = norm_score + tmp

    return norm_score
