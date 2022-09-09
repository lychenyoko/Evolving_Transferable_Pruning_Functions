from Util.Discriminant_Function import GetDI, Abs_SNR, SymmetricDivergence, T_stats_test, FDR, Minibatch_MMD
from Util.GA_Discriminant_Function import GA_Func1, GA_Func2, GA_inspired_Func1, GA_Func3, GA_Func4, GA_inspired_Func2
from Util.mask_util import GetMaskDict, Print_Mask_Shape
import numpy as np
import random

DISCRIMINANT_FUNC_LIST = ['DI','DILoss','AbsSNR', 'SymDiv', 'TwoT', 'FDR', 'MMD', 'GA_Func1', 'GA_Func2', 'GA_Inspired_1', 'GA_Func3', 'GA_Func4', 'GA_Inspired_2', 'Random']
RESNET_56_FULL_SHAPE = [
    [[16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
    [[16, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]],
    [[32, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]
]


def Get_Network_Channel_Score_List(layers, X, y, batch_size, images_pl, mask_list, mask_dict, sess, func = 'DI', Multi_Res = False, y2 = None, block_thres = None, info_print = False):
    '''
    Usage:
        Get the important score of every channel in the network

    Args:
	layers:      A dictionary whose key are the block names and values are dictionaries of the blocks layers tensor
        X:           Samples of input data for caclulating channel score
        y:           Samples of corresponding label for caclulating channel score
        batch_size:  The size of minibatch for getting the channel value
        images_pl:   The input placeholder
        mask_list:   The list of mask indicating each channel's on or off
        mask_dict:   The dictionary whose keys are the mask placeholder and values are the actual mask list
        sess:        The session to run the operation
        func:        The discriminant function for evaluating the channel score, default to be DI
        Multi_Res:   Whether we calculate the channel score based on more than 1 label or not
        y2:          The second label of the whole data, normally coarser than the y
        block_thres: The threshold of the multi-resolution, layer smaller than threshold use coarser label
        info_print:  Whether or not print the channel scoring details 

    Return:
        A list containing the score of each channel in each layer
    '''
 
    print('\n' + '-----------------------------Scoring Channels Now-----------------------------')
    print('Using Multi Resolution: ' + str(Multi_Res) + '\n')

    if Multi_Res:
        assert y2 is not None
        assert block_thres is not None

    Net_score_list = []
    blk_iter_ind = 0
    
    for stage_i in range(len(mask_list)):
        for blk_j in range(len(mask_list[stage_i])):
            blk_keys = list(layers.keys())[blk_iter_ind]
            
            if Multi_Res and (blk_iter_ind <= block_thres):
                y_block = y2
                if info_print:
                    print('\n' + 'Using y2 for channel scoring.')
            else:
                y_block = y
                if info_print:
                    print('Using y for channel scoring.')

            for lay_k in range(len(list(layers[blk_keys].keys()))):

                layer_key = list(layers[blk_keys].keys())[lay_k]
                layer_tensor = layers[blk_keys][layer_key]
                layer_name = blk_keys + ' ' + layer_key
                layer_mask = mask_list[stage_i][blk_j][lay_k]
               
                map_score_list = Get_Layer_Score_List(layer_tensor = layer_tensor, layer_mask = layer_mask, layer_name = layer_name, 
                                                X = X, y = y_block, batch_size = batch_size, func = func,
                                                images_pl = images_pl, mask_dict = mask_dict, sess = sess, info_print = info_print)
                
                Net_score_list.append(map_score_list) 

            blk_iter_ind = blk_iter_ind + 1

    return Net_score_list


def Get_Layer_Score_List(layer_tensor, layer_mask, layer_name, X, y, batch_size, func, images_pl, mask_dict, sess, info_print):
    ''' 
    Usage:
        Get the DI score of every channel in a specific layer

    Args:
	layer_tensor: The tensor of the particular layer
        layer_mask:   The mask of the layer
        layer_name:   The name of the layer, for printing only 
        X:            Samples of input data for caclulating channel score
        y:            Samples of corresponding label for caclulating channel score
        batch_size:   The size of minibatch for getting the channel value
        func:         The discriminant function for evaluating the channel score
        images_pl:    The input placeholder
        mask_dict:    The dictionary whose keys are the mask placeholder and values are the actual mask list
        sess:         The session to run the operation
        info_print:   Whether or not print the channel scoring details 

    Return:
        A list containing the score of each channel in each layer
    '''

    # Decide which discriminant function to use
    assert func in DISCRIMINANT_FUNC_LIST
    if func == 'DI' or func == 'DILoss':
        eval_func = GetDI
    elif func == 'AbsSNR':
        eval_func = Abs_SNR
    elif func == 'SymDiv':
        eval_func = SymmetricDivergence
    elif func == 'TwoT':
        eval_func = T_stats_test
    elif func == 'FDR':
        eval_func = FDR
    elif func == 'MMD':
        eval_func = Minibatch_MMD
    elif func == 'Random':
        eval_func = random.random
    elif func == 'GA_Func1':
        eval_func = GA_Func1
    elif func == 'GA_Func2':
        eval_func = GA_Func2
    elif func == 'GA_Inspired_1':
        eval_func = GA_inspired_Func1
    elif func == 'GA_Func3':
        eval_func = GA_Func3
    elif func == 'GA_Func4':
        eval_func = GA_Func4
    elif func == 'GA_Inspired_2':
        eval_func = GA_inspired_Func2

    if info_print:
        print()
        print(func)
        print(eval_func.__name__)
    
    if func == 'Random':
        map_score_list = [eval_func() for i in range(sum(layer_mask))] 
        
    else: 
        lay_shape = list(layer_tensor.shape)
        lay_shape[0] = X.shape[0]
        lay_out = np.zeros(lay_shape)

        num_batch = len(y)//batch_size
        for i in range(num_batch):
            prune_dict = dict(mask_dict)
            prune_dict[images_pl] = X[batch_size*i:batch_size*(i+1)]
            tmp_out = sess.run(layer_tensor, feed_dict=prune_dict)
            lay_out[batch_size*i:batch_size*(i+1)] = tmp_out
        
        layer_active = lay_out[:,:,:,layer_mask]
        if func == 'DILoss':
            layer_active = np.mean(layer_active, axis = (1,2)) # now such tensor is in dimension of [N, C-1]
        
        if info_print:
            print(layer_name)
            print(layer_active.shape)   
        
        '''Get the Score for active map here'''
        map_score_list = []
        for k in range(layer_active.shape[-1]):
            if func == 'DILoss':
                tmp_mask = [True] * layer_active.shape[-1]
                tmp_mask[k] = False
                fea_map_collection = layer_active[...,tmp_mask]
                map_score_list.append(-eval_func(fea_map_collection, y))

            else:
                fea_map_collection = layer_active[:,:,:,k].reshape(layer_active.shape[0],-1)
                map_score_list.append(eval_func(fea_map_collection, y))   
        
        del lay_out

    if info_print:
        print(map_score_list)
    
    return map_score_list


def Generate_New_Mask_List(layers, Net_Score_List, mask_list, sel_list, rmve_list, info_print = False):
    '''
    Usage:
        Get the new mask by updating the old one with the channel score list and the (selection,removal) list

    '''
    print('\n' + '-----------------------------Actual Pruning Happens-----------------------------')
    
    blk_iter_ind = 0
    layer_iter_ind = 0
    for stage_i in range(len(mask_list)):
        for blk_j in range(len(mask_list[stage_i])):
            blk_keys = list(layers.keys())[blk_iter_ind]
            for lay_k in range(len(list(layers[blk_keys].keys()))):

                layer_key = list(layers[blk_keys].keys())[lay_k]
                layer_name = blk_keys + ' ' + layer_key            
                
                layer_mask = mask_list[stage_i][blk_j][lay_k]
                layer_sel = sel_list[stage_i][blk_j][lay_k]
                layer_rmv = rmve_list[stage_i][blk_j][lay_k]                                
                layer_score_list = Net_Score_List[layer_iter_ind]

                if info_print:
                    print('\n' + 'Layer ID: ' + str(layer_iter_ind))
                    print(layer_name)
                    print('Layer Sel: ' + str(layer_sel))

                # Pruning maps
                if (sum(layer_mask) > layer_rmv and layer_rmv > 0): # Make sure that the active node is larger than the number of node to removed!
                    sel_node = sorted(range(len(layer_score_list)), key= lambda i: layer_score_list[i], reverse=False)[:layer_sel]
                    rmv_node = random.sample(sel_node,layer_rmv)
                    mask_ind = np.where(layer_mask == True)[0][rmv_node]
                    layer_mask[mask_ind] = False

                    print('We have masked out  #' + str(mask_ind) + ' in ' + layer_name + '. It will have ' 
                    + str(sum(layer_mask)) +' nodes/maps.')

                layer_iter_ind = layer_iter_ind + 1

            blk_iter_ind = blk_iter_ind + 1



def Get_Uniform_RmveSel_List(full_network_shape, prune_ratio):
    '''
    Usage:
        Return a remove and select list for all the layer with a pre-specified ratio on the full network shape
    '''
    rmve_list = (np.array(full_network_shape) * prune_ratio).astype(np.int)
    sel_list = rmve_list
    return rmve_list, sel_list





def Get_Layer_Sensitivity(layers, Net_Score_List, prune_list, net_mask_pl, mask_list, sess, acc_score, info_print = False):
    '''
    Usage:
        Get each layer's sensitivity by removing each layer with a certain number of channels and evaluate the then evaluate the acc without re-training.
    
    Args:
        layers:         The dictionary which helps generate the name of a particular layer.
        Net_Score_List: The list which contains each layer's channels scores as a list
        prune_list:     A list containing number of channel to be removed in each layer for "sensitivity" testing
        net_mask_pl:    The mask placeholder
        mask_list:      The current mask of the network
        sess:           The session for running the operation
        acc_score:      The operation to run for tracking the performance
        info_print:     Whether or not to print the intermediate running results
        
    Return:
        A list indicating each layer's name and its sensitivity measured in acc_score
    '''
    
    from copy import deepcopy
    print('\n' + '-----------------------------Conducting Layer Sensitivity Analysis-----------------------------')
    Net_Sen_List = []

    blk_iter_ind = 0
    layer_iter_ind = 0
    
    for stage_i in range(len(mask_list)):
        for blk_j in range(len(mask_list[stage_i])):
            blk_keys = list(layers.keys())[blk_iter_ind]
            for lay_k in range(len(mask_list[stage_i][blk_j])):
                
                layer_key = list(layers[blk_keys].keys())[lay_k]
                layer_name = blk_keys + ' ' + layer_key
                
                if info_print:
                    print('\n' + layer_name + ': ind ' + str(layer_iter_ind))
                
                tmp_mask = deepcopy(mask_list)
                layer_score_list = Net_Score_List[layer_iter_ind]
                num_rmv = prune_list[stage_i][blk_j][lay_k]
                
                if len(layer_score_list) <= num_rmv: # Make sure that the active node is larger than the number of node to removed!
                    mask_acc = 0
                    if info_print:
                        print('''We don't prune!''')
                
                else:
                    rmv_node = sorted( range(len(layer_score_list)), key = lambda k: layer_score_list[k], reverse=False)[:num_rmv]
                    tmp_layer_mask = tmp_mask[stage_i][blk_j][lay_k]
                    mask_ind = np.where(tmp_layer_mask == True)[0][rmv_node]
                    tmp_layer_mask[mask_ind] = False
                    
                    if info_print:
                        print('We will prune: ' + str(mask_ind))
                                        
                    tmp_mask_dict = GetMaskDict(net_mask_pl,tmp_mask)
                    mask_acc = sess.run(acc_score,tmp_mask_dict)
                
                if info_print:
                    print('The temporary mask is: ')
                    Print_Mask_Shape(tmp_mask)
                    print('The test acc is: ' + str(mask_acc))
                    
                Net_Sen_List.append([layer_name, mask_acc])
                layer_iter_ind = layer_iter_ind + 1
            blk_iter_ind = blk_iter_ind + 1
        
    return Net_Sen_List




def Get_Rmve_List(Net_Sen_List, num_rmve_lay, sen_test_list, layers, info_print = False):
    '''
    Usage:
        Get the number of channels to be removed in each layer for the whole network
    
    Args:
        Net_Sen_List: The sensitivity of each layer of the network
        num_rmve_lay: The number of layer to be removed 
        sen_test_list: The number of channel removed in the sensitivity test
        layers: The dictionary containing the layers name info
        info_print: Whether or not to print the intermediate running results
    
    Return:
        The remove list of the whole network
    '''
    
    
    print('\n' + '-----------------------------Generate Pruning List-----------------------------')
    
    # Get the target layer to mask
    lay_priority = sorted(Net_Sen_List, key=lambda a: a[1], reverse=True)
    lay_to_mask = lay_priority[:num_rmve_lay] 
    mask_lay_name = [item[0] for item in lay_to_mask]
    
    if info_print:
        print('\n' + 'The target layer to mask and its mask acc: ')
        print(lay_to_mask)
        
    # Get the remove layer now
    rmve_list = np.array(sen_test_list) * 0 # initialized the rmve list to 0
    
    blk_iter_ind = 0
    for stage_i in range(len(sen_test_list)):
        for blk_j in range(len(sen_test_list[stage_i])):
            blk_keys = list(layers.keys())[blk_iter_ind]
            for lay_k in range(len(sen_test_list[stage_i][blk_j])):
                layer_key = list(layers[blk_keys].keys())[lay_k]
                layer_name = blk_keys + ' ' + layer_key
                
                if layer_name in mask_lay_name:
                    rmve_list[stage_i][blk_j][lay_k] = sen_test_list[stage_i][blk_j][lay_k]
            blk_iter_ind = blk_iter_ind + 1
    return rmve_list

### ================================== Currently Unused Code ===================================

# Helper Function for Map Removing
def Prune_Maps_By_DI(layer_output,y,num_useless_maps,num_map_removed):
    '''Get the DI for Each Map here'''
    map_DI_list = []
    for k in range(layer_output.shape[-1]):
        fea_map_collection = layer_output[:,:,:,k].reshape(layer_output.shape[0],-1)
        map_DI_list.append(GetDI(fea_map_collection,y,DI_rho))

    useless_nodes = sorted(range(len(map_DI_list)), key= lambda i: map_DI_list[i], reverse=False)[:num_useless_maps]
    removed_map = random.sample(useless_nodes,num_map_removed)
    return removed_map


def Print_Prune_Layer(sel_list,rmve_list,num_res_blocks):
    blk_list_name = [32,16,8]
    for blk_list_i in range(Three_Fea_Map_Size):
        for blk_j in range(num_res_blocks):
            for lay_k in range(Two_Mask_Per_Block): 
                lay_sel, lay_rmv = sel_list[blk_list_i][blk_j][lay_k], rmve_list[blk_list_i][blk_j][lay_k]
                if lay_rmv == 0:
                    continue
                lay_name = 'Res' + str(blk_list_name[blk_list_i]) + '_' + str(blk_j) + '_' + str(lay_k + 1)
                print('  ' + lay_name + '_Sel_Rmv: ' + str((lay_sel,lay_rmv)) )
    print(' ')




