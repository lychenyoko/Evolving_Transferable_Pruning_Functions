import numpy as np
import random

import pickle as pkl
import os
import datetime

FLOP_38   = 57369600
param_38  = 392640

FLOP_56   = 125485696
param_56  = 856816

FLOP_164  = 244540416
param_164 = 1709760

FLOP_110  = 252887680
param_110 = 1735792

def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
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


# ------ Logs and Model Saving Part ------
def LogTraining(saved_network,log_stats,best_acc,FLOP,param,num_layer):

    if num_layer == 38:
        FLOP_save  = str(round(FLOP/FLOP_38 * 100,2))
        param_save = str(round(param/param_38 * 100,2))       

    if num_layer == 56: # Normal block
        FLOP_save  = str(round(FLOP/FLOP_56 * 100,2))
        param_save = str(round(param/param_56 * 100,2))

    if num_layer == 110:
        FLOP_save  = str(round(FLOP/FLOP_110 * 100,2))
        param_save = str(round(param/param_110 * 100,2))

    if num_layer == 164: # BottleNeck block
        FLOP_save  = str(round(FLOP/ FLOP_164 * 100,2))
        param_save = str(round(param/param_164 * 100,2))
    

    m_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    file_name_str = 'ResNet' + str(num_layer) + '_' + m_time + '_acc_' + str(round(best_acc * 100,2)) + \
                '_FLOP_' + FLOP_save + '_Param_' + param_save
    dir_name = './' + file_name_str
    os.mkdir(dir_name)
    
    log_file = dir_name + '/TF_log_' + m_time + '.pkl'
    pkl.dump(log_stats, open(log_file, "wb")) # Save training log
    
    model_file = dir_name + '/' + file_name_str + '.npy'
    print('The file name is: ' + model_file)
    np.save(model_file, saved_network)  


def DataSampling(X, y, num_sample, balanced):
    '''
    Usage:
        Sample a subset of data, the subset can be either balanced or unbalanced
        
    Args:
        X: (np.array) of images with shape [N, H, W, C]
        y: (np.array) of corresponding labels with shape [N, 1] or [N]
        num_sample: (int) the number of images to be sampled
        balanced: (bool) whether the subsampled dataset should be balanced or not 
    '''
    
    if balanced:
        label_array = np.unique(y)
        num_sample_per_class = num_sample // len(label_array)
        
        sel_indices = []
        for label in label_array:
            label_sel_indices = np.random.permutation(np.where(y == label)[0])[:num_sample_per_class]
            sel_indices += list(label_sel_indices)
        
        sel_indices = np.array(sel_indices)
        np.random.shuffle(sel_indices)
               
    else:
        sel_indices = np.random.permutation(np.arange(len(y)))[:num_sample]

    assert len(np.unique(sel_indices)) == num_sample # Asserting the indices are all uniques
    sampled_X = X[sel_indices, ...]
    sampled_y = y[sel_indices]
    
    return sampled_X, sampled_y
