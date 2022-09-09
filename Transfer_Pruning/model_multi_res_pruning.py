import numpy as np
import tensorflow as tf
from tensorflow import keras

import time

from Net_Structure.ResNet import ResNet
from Util.training_util import augment_all_images, LogTraining
from Util.Calculators import ResNetParamCal, ResNetFLOPCal, Print_Network, Get_FLOP_Reduction_Per_Layer, ResNetFLOPCal_Numpy
from Util.mask_util import GetDefaultMaskList, GetMaskDict, Print_Mask_Shape
from Util.pruning_augmenting_util import Get_Network_Channel_Score_List, Generate_New_Mask_List, Get_Uniform_RmveSel_List
from Util.pruning_augmenting_util import Get_Layer_Sensitivity, Get_Rmve_List

'''-------Load Hyper-parameters and Set Them to Global Parameters------'''
from ResNet_hyperparams_multi_res_pruning import *
if block_type == 'Normal':
    num_layer = n_res_block * 6 + 2
if block_type == 'BottleNeck':
    num_layer = n_res_block * 9 + 2

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--lr_drop_epoch',type = int)
parser.add_argument('--lr_drop_rate',type = float)
parser.add_argument('--ridge',type = float)
parser.add_argument('--model',type = str)
parser.add_argument('--init_lr', type = float)
parser.add_argument('--opt', type = str)
parser.add_argument('--bn_momentum', type = float)
parser.add_argument('--Pfunc', type = str)
parser.add_argument('--prune_ratio', type = float)
parser.add_argument('--pruning_mode', type = str)
parser.add_argument('--prune_step', type = int)
parser.add_argument('--prune_multiple', type = float)
parser.add_argument('--num_rmve_lay', type = int)
parser.add_argument('--front_label', type = str)
parser.add_argument('--rear_label', type = str)
parser.add_argument('--coarse_label', type = str)
parser.add_argument('--block_threshold', type = int)
args = parser.parse_args()

if args.lr_drop_epoch is not None:
    lr_drop_epoch = args.lr_drop_epoch
if args.lr_drop_rate is not None:
    lr_drop_rate = args.lr_drop_rate
if args.ridge is not None:
    m_ridge = args.ridge
if args.model is not None:
    model = args.model
if args.init_lr is not None:
    init_lr = args.init_lr 
if args.opt is not None:
    opt = args.opt
if args.bn_momentum is not None:
    bn_momentum = args.bn_momentum
if args.Pfunc is not None:
    Pfunc = args.Pfunc
if args.prune_ratio is not None:
    prune_ratio = args.prune_ratio
if args.pruning_mode is not None:
    pruning_mode = args.pruning_mode
if args.prune_step is not None:
    prune_step = args.prune_step
if args.prune_multiple is not None:
    prune_multiple = args.prune_multiple
if args.num_rmve_lay is not None:
    num_rmve_lay = args.num_rmve_lay
if args.front_label is not None:
    front_label = args.front_label
if args.rear_label is not None:
    rear_label = args.rear_label
if args.coarse_label is not None:
    cifar_ytr_coarse_file = args.coarse_label
if args.block_threshold is not None:
    block_threshold = args.block_threshold

def LoadCIFAR(Dataset):
    assert Dataset in ['CIFAR100', 'CIFAR10']
    if Dataset == 'CIFAR100':                
        dataset = keras.datasets.cifar100
    elif Dataset == 'CIFAR10':
        dataset = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    mean = 120.707
    std = 64.15
    Xtr = (train_images - mean)/(std + 1e-7)
    Xte = (test_images - mean)/(std + 1e-7) # The way for data standardization a bit different from MNIST
    ytr = train_labels
    yte = test_labels
    return Xtr, Xte, ytr, yte

def GetTrainGraph():
    # Tell TensorFlow that the model will be built into the default Graph.
    tf.reset_default_graph()
    if model_loaded:
        model_value = np.load(model, allow_pickle = True)
        mResNet = ResNet(param_list = list(model_value), block_type = block_type, num_res_blocks = n_res_block, BN_Var_Epsilon = bn_var_epsilon, BN_Momentum = bn_momentum)
    else:
        mResNet = ResNet(block_type = block_type, num_res_blocks = n_res_block, BN_Var_Epsilon = bn_var_epsilon, BN_Momentum = bn_momentum)

    return mResNet


def GetTrainOp(mResNet):
    '''-------------- Training Ops --------------'''
    # Generate placeholders for the images and labels.    
    images_pl = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS,IMAGE_PIXELS,IMAGE_CHANNEL))
    labels_pl = tf.placeholder(tf.int64, shape=(None))
    net_mask_pl = mResNet.net_get_mask_pl_list()

    # Build a Graph that computes predictions from the inference model.
    layers, logits = mResNet.inference(images_pl, is_training = True, res_mask_list = net_mask_pl)
    
    # Add to the Graph the Ops for loss calculation.
    entropy_loss, ridge_loss = mResNet.loss(logits, labels_pl, m_ridge)
    training_loss = entropy_loss + ridge_loss
    
    # Add to the Graph the Ops that calculate and apply gradients.
    lr_pl = tf.placeholder(tf.float32)
    train_op = mResNet.training(opt, training_loss, lr_pl, momentum = opt_momentum)
     
    return images_pl, labels_pl, net_mask_pl, lr_pl, layers, train_op


def GetTestScore(mResNet, Xte, yte, net_mask_pl):
    dummy, test_logits = mResNet.inference(Xte.astype(np.float32), is_training = False, res_mask_list = net_mask_pl)
    correct_prediction = tf.equal(tf.argmax(test_logits, 1), yte.reshape(-1))
    acc_score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
    return acc_score   


def PrintExperimentStatus():
    '''
    Print the hyper-parameters for training and pruning
    '''
    
    print('\n' + '----------- Training Start -----------' + '\n')
    print('Training Params: ' + '\n' + '\n' +

          '  Dataset and Model: ' + '\n' + 
          '    Dataset: ' + Dataset + '\n' + 
          '    Block Type: ' + block_type + '\n' +
          '    Num Res Blocks: ' + str(n_res_block) + '\n' +
          '    Num Layers: ' + str(num_layer) + '\n' + '\n' + 

          '  Optimization Scheme: ' + '\n' + 
          '    Epochs: ' + str(MAX_EPOCH) + '\n' +
          '    Batch Size: ' + str(Batch_Size) + '\n' +         
          '    Optimizer: ' + opt + '\n' +
          '    Opt Momentum: ' + str(opt_momentum) + '\n' +          
          '    Init LR: ' + str(init_lr) + '\n' +
          '    LR Drop Epoch: ' + str(lr_drop_epoch) + '\n' + 
          '    LR Drop Rate: ' + str(lr_drop_rate) + '\n' +  '\n' +

          '  Regularization Scheme: ' + '\n' + 
          '    Ridge: ' + str(m_ridge) + '\n' +       
          '    Data Augmentation: ' + str(use_data_aug) + '\n' +
          '    BN Momentum: ' + str(bn_momentum) + '\n' +
          '    BN Var Epsilon: ' + str(bn_var_epsilon) + '\n'  + '\n' +
      
          'Pruning Params: ' + '\n' + 
          '  Pruning Mode: ' + pruning_mode + '\n' +
          '  Pruning Functions: ' + Pfunc + '\n' + 
          '  Pruning Ratio: ' + str(prune_ratio) + '\n' + 
          '  Pruning Multiple: ' + str(prune_multiple) + '\n' + 
          '  Pruned Layer: ' + str(num_rmve_lay) + '\n' +
          '  Pruning Step: ' + str(prune_step) + '\n' + '\n' +

          'Multi Resolution Params: ' + '\n' +
          '  Coarse Label: ' + str(cifar_ytr_coarse_file) + '\n' +
          '  Front Label: ' + str(front_label) + '\n' + 
          '  Rear Label: ' + str(rear_label) + '\n' +
          '  Block Threshold: ' + str(block_threshold) + '\n'
    )
    
    if model_loaded:
        print('Init Model:' + '\n'
              '  ' + str(model) + '\n'
    )       

def GetTrainParam():
    train_params = {'Epochs': MAX_EPOCH, 'Batch Size':Batch_Size, 
                    'Init LR':init_lr,  'LR Drop Epoch':lr_drop_epoch, 'LR Drop Rate': lr_drop_rate, 
                    'Optimizer':opt, 'Opt Momentum':opt_momentum,
                    'Ridge': m_ridge, 'Data Aug': use_data_aug, 'BN Momentum': bn_momentum, 'BN Var Epsilon': bn_var_epsilon,
                    'Pfunc': Pfunc, 'Pratio': prune_ratio, 'Pmode': pruning_mode, 'Player': num_rmve_lay}
    return train_params

def PrintRmveList(rmve_list):
    for stage_rmve in rmve_list:
        print([list(blk_rmv) for blk_rmv in stage_rmve])


def EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network, mask_dict):
    # --------------------------------------- Evaluation Step ---------------------------------------
    te_acc = sess.run(acc_score, feed_dict = mask_dict)
    te_acc_list.append(te_acc)
    
    if te_acc >= best_acc:
        best_acc = te_acc
        saved_network = mResNet.model_save(sess)
    
    if te_acc >= epoch_best_acc:
        epoch_best_acc = te_acc
    
    return best_acc, epoch_best_acc, saved_network

def TrainStep(ind, iter_per_epoch, shuf_Xtr, shuf_ytr, current_lr, images_pl, labels_pl, lr_pl, train_op, mask_dict, sess):
    if (ind != (iter_per_epoch - 1)):
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:(ind+1)*Batch_Size,:],shuf_ytr[ind*Batch_Size:(ind+1)*Batch_Size,:]
    else:
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:,:],shuf_ytr[ind*Batch_Size:,:]
    if use_data_aug:
        mb_x = augment_all_images(mb_x,4)

    train_dict = dict(mask_dict)
    train_dict[images_pl] = mb_x
    train_dict[labels_pl] = mb_y
    train_dict[lr_pl] = current_lr

    _ = sess.run(train_op,train_dict)

def FinalNetworkEvaluation(mResNet, saved_network, mask_list, Xte, yte, log_stats, sess):
    mResNet.model_load(saved_network, sess)
    comp_param = mResNet.compressed_model(mask_list,sess)
    
    compResNet = ResNet(param_list = list(comp_param), block_type = block_type, num_res_blocks = n_res_block, BN_Var_Epsilon = bn_var_epsilon, BN_Momentum = bn_momentum)
    dummy, comp_logits = compResNet.inference(Xte.astype(np.float32))
    comp_pred = tf.equal(tf.argmax(comp_logits, 1), yte.reshape(-1))
    comp_acc = tf.reduce_mean(tf.cast(comp_pred, tf.float32), name = 'final_acc')
    
    compResNet.model_initialization(sess)
    final_acc = sess.run(comp_acc)
    final_FLOP = ResNetFLOPCal(compResNet)
    final_param = ResNetParamCal(compResNet)

    print('\n' + 'The final network shape is: ')
    Print_Network(compResNet)
     
    LogTraining(saved_network = comp_param, log_stats = log_stats,
                best_acc = final_acc, FLOP = final_FLOP, param = final_param, num_layer = num_layer)

def main():
    '''
    Main method of the script
    '''
    Xtr, Xte, ytr, yte = LoadCIFAR(Dataset) # Dataset Loading
    ytr_coarse = np.load(cifar_ytr_coarse_file, allow_pickle = True)
    if front_label == 'coarse':
        y_front = ytr_coarse
    elif front_label == 'fine':
        y_front = ytr

    if rear_label == 'coarse':
        y_rear = ytr_coarse
    elif rear_label == 'fine':
        y_rear = ytr

    '''-------Training Graph and Operation------'''
    mResNet = GetTrainGraph()
    images_pl, labels_pl, net_mask_pl, lr_pl, layers, train_op = GetTrainOp(mResNet)
        
    '''-------Testing Acc Tensor -------'''
    acc_score = GetTestScore(mResNet, Xte, yte, net_mask_pl)
  
    # Initialize the Training Grpah
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Start time counter 
    start_time = time.time()

    ##--------------------------------- Experiment Begins ----------------------------------
    PrintExperimentStatus()
    train_params = GetTrainParam()
    mask_list = GetDefaultMaskList(net_mask_pl)

    for step in range(prune_step):
    
        print('\n' + '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Entering Pruning Step: ' + str(step) + '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
        mask_dict = GetMaskDict(net_mask_pl,mask_list)
        cur_acc = sess.run(acc_score, feed_dict = mask_dict)
        print('Current Accuracy is: ' + str(cur_acc))
        
        # Pruning Process >>>
        # Get the score for each channel in the network
        Net_Score_List = Get_Network_Channel_Score_List(
            layers = layers, X = Xtr, y = y_rear, batch_size = 2000, images_pl = images_pl,
            mask_list = mask_list, mask_dict = mask_dict, sess = sess, func = Pfunc, 
            Multi_Res = True, y2 = y_front, block_thres = block_threshold, info_print = True
        )

        # Get the remove/sel list for two pruning conditions
        assert pruning_mode in PRUNING_MODE
        if pruning_mode == 'Uniform':
            rmve_list, sel_list = Get_Uniform_RmveSel_List(full_network_shape = full_network_shape, prune_ratio = prune_ratio)
        elif pruning_mode == 'Predefined':
            rmve_list, sel_list = pre_rmve_list, pre_sel_list
        elif (pruning_mode == 'Automatic_Ratio') or (pruning_mode == 'Automatic_FLOPs'):
            
            if pruning_mode == 'Automatic_Ratio':
                sen_prune_list, _ = Get_Uniform_RmveSel_List(full_network_shape = full_network_shape, prune_ratio = prune_ratio)

            elif pruning_mode == 'Automatic_FLOPs':
                cur_FLOP = ResNetFLOPCal_Numpy(mResNet.compressed_model(mask_list=mask_list, m_sess = sess), num_res_block=mResNet.num_res_blocks)
                net_reduced_FLOP = Get_FLOP_Reduction_Per_Layer(mask_list = mask_list, ResNet_Obj=mResNet, sess = sess, ori_FLOP = cur_FLOP, info_print = True)
                sen_prune_list = ( np.max(net_reduced_FLOP) / np.array(net_reduced_FLOP) * prune_multiple ).astype(int)                
            
            Net_Sen_List =  Get_Layer_Sensitivity(
                Net_Score_List=Net_Score_List, mask_list=mask_list, prune_list=sen_prune_list,
                net_mask_pl=net_mask_pl, sess=sess, layers=layers, acc_score=acc_score,
                info_print=True)
           
            rmve_list = Get_Rmve_List(Net_Sen_List=Net_Sen_List, num_rmve_lay=num_rmve_lay, 
                                  layers=layers, sen_test_list=sen_prune_list, info_print=True)
             
            sel_list = rmve_list

        # Print the rmve_list
        print('\n' + 'The Remove List is: ')
        PrintRmveList(rmve_list)

        # Get the new mask after pruning
        Generate_New_Mask_List( Net_Score_List = Net_Score_List,
            layers = layers, mask_list = mask_list,
            sel_list = sel_list, rmve_list = rmve_list, info_print = True
        )

        # Check the shape of new mask and update mask dict
        print('\n' + 'Current Network Shape: ')
        Print_Mask_Shape(mask_list)
        print()
        updated_mask_dict = GetMaskDict(net_mask_pl, mask_list)     
        # <<< Pruning Process


        # >>> Re-Training Process
        te_acc_list = []
        best_acc = 0  
        saved_network = mResNet.model_save(sess)
        iter_per_epoch = Xtr.shape[0] // Batch_Size
   
        for epoch in range(MAX_EPOCH):
        
            shuf_ind = list(range(Xtr.shape[0]))
            np.random.shuffle(shuf_ind)
            shuf_Xtr,shuf_ytr = Xtr[shuf_ind,:],ytr[shuf_ind]
            
            epoch_start = time.time()
            epoch_best_acc = 0
            current_lr = init_lr * (lr_drop_rate ** (epoch // lr_drop_epoch))
            
            best_acc, epoch_best_acc, saved_network = EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network, updated_mask_dict)
        
            # --------------------------------------- Training Step ---------------------------------------
            for ind in range(iter_per_epoch):
                TrainStep(ind, iter_per_epoch, shuf_Xtr, shuf_ytr, current_lr, images_pl, labels_pl, lr_pl, train_op, updated_mask_dict, sess)
        
                # --------------------------------------- Evaluation Step ---------------------------------------
                if (epoch >= 150) and (ind % 30 == 29):
                    best_acc, epoch_best_acc, saved_network = EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network, updated_mask_dict)    
        
            # Epoch Status Printing
            epoch_end = time.time()
            print('Epoch #' + str(epoch) + ' Takes time: ' + str(round(epoch_end - epoch_start,2)) + ' Epoch Best Acc: ' + str(epoch_best_acc) +
                ' Best Acc: ' + str(best_acc) + ' Current Lr: ' + str(current_lr))
   
        end_time = time.time()
        print('\n' + 'Total Time For Training: ' + str(end_time - start_time))
            
        log_stats = {'testing acc': te_acc_list, 'params':train_params}
        FinalNetworkEvaluation(mResNet, saved_network, mask_list, Xte, yte, log_stats, sess)

if __name__ == '__main__':
    main()    
