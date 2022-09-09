import numpy as np
import random
import datetime
import os
import subprocess as cmd

def Prune_Nodes_Channel_By_Func(layer_tensor,label,eval_func,num_useless_node,num_node_removed,ker):
    '''
    Usage: 
        To return the index channel/nodes to be removed for LeNet5
    '''
    score_list = []
    for k in range(layer_tensor.shape[-1]):
        fea_k = layer_tensor[...,k].reshape(layer_tensor.shape[0],-1)
        score_list.append(eval_func(fea_k,label,ker[...,k],ker))

    # randomly pick num_node_removed from the least num_useless_node nodes
    useless_nodes = sorted(range(len(score_list)), key= lambda i: score_list[i], reverse=False)[:num_useless_node]
    removed_node = random.sample(useless_nodes,num_node_removed)
    return removed_node  


def mask_fla_from_conv2(_conv2,fla_sel_true,_fla):
    for i in range(len(_conv2)):
        if _conv2[i] == False:
            fla_ind = np.where(np.logical_and(fla_sel_true >= (i * 16), fla_sel_true < (i+1) * 16))[0]
            tmp = np.array(_fla)
            tmp[fla_ind] = False
            _fla = list(tmp)
    return np.array(_fla)

def LeNet5_Model_Save(model, sess, acc, FLOP, Param, genid, indid, saving_dir):
    network = model.model_save(sess)
    m_fla_sel = model.fla_sel
    
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    m_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = 'LeNet5_Ind_' + str(indid) + '_' + m_time + '_acc_' + str(round(acc * 100,2)) + \
                '_FLOP_' + str(round(FLOP*100, 2)) + '_Param_' + str(round(Param*100, 2))
    print('The filename is: ' + file_name)
    np.save(saving_dir + file_name +'.npy', [network,m_fla_sel])

def VGG16_Model_Save(model, acc, FLOP, Param, genid, indid, saving_dir):
    
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    m_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = 'VGG16_Ind_' + str(indid) + '_' + m_time + '_acc_' + str(round(acc * 100,2)) + \
                '_FLOP_' + str(round(FLOP*100, 2)) + '_Param_' + str(round(Param*100, 2)) 
    print('The filename is: ' + file_name)
    np.save(saving_dir + file_name +'.npy', model)


# Helper function for data augmentation
def augment_image(image, pad):
    '''
    Usage:
        Randomly perform zero padding, randomly crop image to original size and mirror horizontally
    '''
    flip = random.getrandbits(1)
    if flip:
        image = image[:, ::-1, :]
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    return cropped


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images
