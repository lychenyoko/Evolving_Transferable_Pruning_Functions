import numpy as np
import tensorflow as tf
from tensorflow import keras

import time
from collections import defaultdict

from Net_Structure.ResNet import ResNet
from Util.training_util import augment_all_images, LogTraining
from Util.hierarchy_util import Generate_Coarse_Label
from Util.DCA_util import DCA
from Util.Calculators import ResNetParamCal, ResNetFLOPCal, Print_Network, Get_FLOP_Reduction_Per_Layer, ResNetFLOPCal_Numpy
from Util.mask_util import GetDefaultMaskList, GetMaskDict, Print_Mask_Shape
from Util.pruning_augmenting_util import Get_Network_Channel_Score_List, Generate_New_Mask_List, Get_Uniform_RmveSel_List
from Util.pruning_augmenting_util import Get_Layer_Sensitivity, Get_Rmve_List

'''-------Load Hyper-parameters and Set Them to Global Parameters------'''
from ResNet_hyperparams_kd_investigation import *
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
parser.add_argument('--output_kd_lambda', type = float)
parser.add_argument('--inter_kd_lambda', type = float)
parser.add_argument('--dca_mode', type = str)
parser.add_argument('--dca_learn_freq', type = int)
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
if args.output_kd_lambda is not None:
    output_kd_lambda = args.output_kd_lambda
if args.inter_kd_lambda is not None:
    inter_kd_lambda = args.inter_kd_lambda
if args.dca_mode is not None:
    dca_mode = args.dca_mode
if args.dca_learn_freq is not None:
    dca_learn_freq = args.dca_learn_freq

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

def BuildCoarseFineDict(y_fine, y_coarse):
    '''
    Usage:
        Build a (dict) (a map) with key to be the coarse label and value to be a (set) of fine label
    '''
    coarse_fine_dict = defaultdict(set)
    for y_f,y_c in zip(y_fine,y_coarse):
        coarse_fine_dict[int(y_c)].add(int(y_f))
    return coarse_fine_dict


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
    
    # Add to the Graph the Ops that calculate and apply gradients.
    lr_pl = tf.placeholder(tf.float32)
     
    return images_pl, labels_pl, net_mask_pl, lr_pl, layers, logits, entropy_loss, ridge_loss


def GetTestScore(mResNet, Xte, yte, net_mask_pl):
    dummy, test_logits = mResNet.inference(Xte.astype(np.float32), is_training = False, res_mask_list = net_mask_pl)
    correct_prediction = tf.equal(tf.argmax(test_logits, 1), yte.reshape(-1))
    acc_score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
    return acc_score   

def GetTeacherModel():
    teacher_model_param = np.load(teacher_model, allow_pickle = True)
    teacher_ResNet = ResNet(param_list = list(teacher_model_param), block_type = block_type, num_res_blocks = n_res_block, BN_Var_Epsilon = bn_var_epsilon, BN_Momentum = bn_momentum)
    teacher_images_pl = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS,IMAGE_PIXELS,IMAGE_CHANNEL))
    teacher_layers, teacher_logits = teacher_ResNet.inference(teacher_images_pl)
    teacher_target = tf.nn.softmax(teacher_logits)
    return teacher_ResNet, teacher_images_pl, teacher_layers, teacher_target


def GetHintInterKD(layers, teacher_layers):
    '''
    Usage:
        Create the tensor for hint-based intermediate KD
    '''
    student_inter_tensor, teacher_inter_tensor = layers[inter_kd_block][inter_kd_layer], teacher_layers[inter_kd_block][inter_kd_layer]
    inter_conv_tensor = tf.Variable(tf.truncated_normal([1, 1, int(student_inter_tensor.shape[-1]), int(teacher_inter_tensor.shape[-1])], stddev=0.1,name='inter_conv_tensor'))
    student_transformed_inter_tensor = tf.nn.conv2d(student_inter_tensor, inter_conv_tensor, strides=[1, 1, 1, 1], padding='SAME')
    assert ((np.array(student_transformed_inter_tensor.shape) == np.array(teacher_inter_tensor.shape))[1:]).all()
    teacher_inter_tensor_pl = tf.placeholder(tf.float32, teacher_inter_tensor.shape, name='teacher_inter')
    inter_kd_loss = inter_kd_lambda * tf.reduce_mean(tf.squared_difference(student_transformed_inter_tensor, teacher_inter_tensor_pl))
    return teacher_inter_tensor_pl, inter_kd_loss


def GetDCAInterKD(layers, teacher_layers, y):
    '''
    Usage:
        Create the tensor for hint-based intermediate KD
    '''
    student_inter_tensor, teacher_inter_tensor = layers[inter_kd_block][inter_kd_layer], teacher_layers[inter_kd_block][inter_kd_layer]
    student_vec_dim, teacher_vec_dim =  int(np.prod(np.array(student_inter_tensor.shape)[1:])), int(np.prod(np.array(teacher_inter_tensor.shape)[1:]))
    student_inter_vec, teacher_inter_vec = tf.reshape(student_inter_tensor, shape=(-1, student_vec_dim)), tf.reshape(teacher_inter_tensor, shape=(-1, teacher_vec_dim))

    num_labels = len(np.unique(y))

    teacher_dca_pl = tf.placeholder(tf.float32, [teacher_vec_dim, num_labels], name='student_dca_pl')
    teacher_trans = tf.matmul(teacher_inter_vec, teacher_dca_pl)
    teacher_trans_pl = tf.placeholder(tf.float32, teacher_trans.shape, name='teacher_transform')

    student_dca_pl = tf.placeholder(tf.float32, [student_vec_dim, num_labels], name='student_dca_pl')
    student_trans = tf.matmul(student_inter_vec, student_dca_pl)

    inter_kd_loss = inter_kd_lambda * tf.reduce_mean(tf.squared_difference(student_trans, teacher_trans_pl))
    return student_inter_vec, teacher_inter_vec, teacher_dca_pl, teacher_trans, teacher_trans_pl, student_dca_pl, inter_kd_loss 



def PrintExperimentStatus():
    '''
    Print the hyper-parameters for training and pruning
    '''
    
    print('\n' + '----------- Training Start -----------' + '\n')
    print('Training Params: ' + '\n\n' +

          '  Dataset and Model: ' + '\n' + 
          '    Dataset: ' + Dataset + '\n' + 
          '    Block Type: ' + block_type + '\n' +
          '    Num Res Blocks: ' + str(n_res_block) + '\n' +
          '    Num Layers: ' + str(num_layer) + '\n\n' + 

          '  Optimization Scheme: ' + '\n' + 
          '    Epochs: ' + str(MAX_EPOCH) + '\n' +
          '    Batch Size: ' + str(Batch_Size) + '\n' +         
          '    Optimizer: ' + opt + '\n' +
          '    Opt Momentum: ' + str(opt_momentum) + '\n' +          
          '    Init LR: ' + str(init_lr) + '\n' +
          '    LR Drop Epoch: ' + str(lr_drop_epoch) + '\n' + 
          '    LR Drop Rate: ' + str(lr_drop_rate) + '\n\n' +

          '  Regularization Scheme: ' + '\n' + 
          '    Ridge: ' + str(m_ridge) + '\n' +       
          '    Data Augmentation: ' + str(use_data_aug) + '\n' +
          '    BN Momentum: ' + str(bn_momentum) + '\n' +
          '    BN Var Epsilon: ' + str(bn_var_epsilon) + '\n\n' +
      
          'Pruning Params: ' + '\n' + 
          '  Need Pruning: ' + str(need_pruning) + '\n' +
          '  Pruning Mode: ' + str(pruning_mode) + '\n' +
          '  Pruning Functions: ' + str(Pfunc) + '\n' + 
          '  Pruning Ratio: ' + str(prune_ratio) + '\n' + 
          '  Pruning Multiple: ' + str(prune_multiple) + '\n' + 
          '  Pruned Layer: ' + str(num_rmve_lay) + '\n' +
          '  Pruning Step: ' + str(prune_step) + '\n\n' +

          'Multi Resolution Params: ' + '\n' +
          '  Coarse Label: ' + str(cifar_ytr_coarse_file) + '\n' +
          '  Front Label: ' + str(front_label) + '\n' + 
          '  Rear Label: ' + str(rear_label) + '\n' +
          '  Block Threshold: ' + str(block_threshold) + '\n\n' +

          'Knowledge Distillation Params: ' + '\n' +
          '  Teacher Model: ' + str(teacher_model) + '\n' + 
          '  Output KD Lambda: ' + str(output_kd_lambda) + '\n' +
          '  Inter KD Mode: ' + inter_kd_mode + '\n' + 
          '  DCA Mode: ' + dca_mode + '\n' + 
          '  DCA Learning Frequency (Epochs): ' + str(dca_learn_freq) + '\n' + 
          '  Inter KD Block: ' + inter_kd_block + '\n' + 
          '  Inter KD Layer: ' + inter_kd_layer + '\n' + 
          '  Inter KD Lambda: ' + str(inter_kd_lambda) + '\n' 
    )
    
    if model_loaded:
        print('Init Student Model: ' + str(model) + '\n'
    )
    else:
        print('Student Model Randomly Initialized. ')       

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

def Learn_DCA(images_pl, X, inter_vec_tensor, y, sess, DCA, mask_dict=None):
    '''
    Usage:
        Based on images and labels, learn the DCA components
    '''
    if mask_dict == None:
        feed_dict = {images_pl:X}
    else:
        feed_dict = dict(mask_dict)
        feed_dict[images_pl] = X
    inter_vec_np = sess.run(inter_vec_tensor, feed_dict)
    DCA.fit(inter_vec_np, y)
    num_labels = len(np.unique(y))
    DCA_weights = DCA.components[:num_labels].T
    return DCA_weights    


def NetworkPruning(Xtr, y_rear, y_front, images_pl, layers, mask_list, mask_dict, sess):
    '''
    Usage:
        Update the network topology which is specified by mask_list
    '''

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
    return mask_list


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

def Get_Batch(ind, iter_per_epoch, shuf_Xtr, shuf_ytr):
    '''
    Usage:
        Get the batch of data and label
    '''
    if (ind != (iter_per_epoch - 1)):
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:(ind+1)*Batch_Size,:],shuf_ytr[ind*Batch_Size:(ind+1)*Batch_Size,:]
    else:
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:,:],shuf_ytr[ind*Batch_Size:,:]
    if use_data_aug:
        mb_x = augment_all_images(mb_x,4)
    return mb_x, mb_y

def Get_Teacher_Outputs(teacher_images_pl, teacher_target, teacher_inter_tensor, mb_x, sess):
    '''
    Usage:
        Obtain the teacher's target and layer output
    '''
    teacher_inter, teacher_label = sess.run([teacher_inter_tensor, teacher_target], {teacher_images_pl: mb_x})
    return teacher_inter, teacher_label


def TrainStep(key_val_list, mask_dict, train_op, sess, entropy_loss, ridge_loss, output_kd_loss, inter_kd_loss):

    train_dict = dict(mask_dict)
    for key,val in key_val_list:
        train_dict[key] = val

    _, entropy_loss_val, ridge_loss_val, output_kd_loss_val, inter_kd_loss_val = sess.run([train_op, entropy_loss, ridge_loss, output_kd_loss, inter_kd_loss], train_dict)
    return entropy_loss_val, ridge_loss_val, output_kd_loss_val, inter_kd_loss_val

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
    Xtr, Xte, ytr, yte = LoadCIFAR(Dataset) # Dataset Loading

    # Handling the coarse and fine labels of the dataset
    ytr_coarse = np.load(cifar_ytr_coarse_file, allow_pickle = True)
    coarse_fine_dict = BuildCoarseFineDict(ytr, ytr_coarse)

    if front_label == 'coarse':
        y_front = ytr_coarse
    elif front_label == 'fine':
        y_front = ytr

    if rear_label == 'coarse':
        y_rear = ytr_coarse
    elif rear_label == 'fine':
        y_rear = ytr

    '''-------Training Graph and Operation------'''
    # Student Network
    mResNet = GetTrainGraph()
    images_pl, labels_pl, net_mask_pl, lr_pl, layers, logits, entropy_loss, ridge_loss = GetTrainOp(mResNet)

    # Teacher Network
    teacher_ResNet, teacher_images_pl, teacher_layers, teacher_target = GetTeacherModel()
    
    # Output Knowledge Distillation Loss
    soft_target_pl = tf.placeholder(tf.float32, [None, 100], name='soft_label') # Hard Coded for now
    output_kd_loss = output_kd_lambda * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = soft_target_pl, logits = logits))

    # Intermediate Knowledge Distillation
    assert inter_kd_mode in INTERMEDIATE_DISTILLATION_MODE
    if inter_kd_mode == 'Hint': # Hint-Based Loss
        teacher_inter_tensor_pl, inter_kd_loss = GetHintInterKD(layers, teacher_layers)
    elif inter_kd_mode == 'DCA': 
        if dca_mode == 'coarse':
            y_dca = Generate_Coarse_Label(coarse_fine_dict, yte)
        elif dca_mode == 'fine':
            y_dca = yte
        student_inter_vec, teacher_inter_vec, teacher_dca_pl, teacher_trans, teacher_trans_pl, student_dca_pl, inter_kd_loss = GetDCAInterKD(layers, teacher_layers, y_dca)
        teacher_DCA, student_DCA = DCA(), DCA()    
        

    # Total Loss and Gradient Descent
    total_loss = entropy_loss + ridge_loss + output_kd_loss + inter_kd_loss
    train_op = mResNet.training(opt, total_loss, lr_pl, momentum = opt_momentum)
        
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
    if inter_kd_mode == 'DCA':        
        print('Learning Teacher DCA ...')              
        teacher_DCA_weights = Learn_DCA(teacher_images_pl, Xte, teacher_inter_vec, y_dca, sess, teacher_DCA)

    for step in range(prune_step):
    
        print('\n' + '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Entering Pruning Step: ' + str(step) + '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
        mask_dict = GetMaskDict(net_mask_pl,mask_list)
        cur_acc = sess.run(acc_score, feed_dict = mask_dict)
        print('Current Accuracy is: ' + str(cur_acc))
        
        # Network Pruning 
        if need_pruning:
            mask_list = NetworkPruning(Xtr, y_rear, y_front, images_pl, layers, mask_list, mask_dict, sess)

        # Check the shape of new mask and update mask dict
        print('\n' + 'Current Network Shape: ')
        Print_Mask_Shape(mask_list)
        print()
        updated_mask_dict = GetMaskDict(net_mask_pl, mask_list)     

        # >>> Re-Training Process
        te_acc_list = []
        best_acc = 0  
        saved_network = mResNet.model_save(sess)
        iter_per_epoch = Xtr.shape[0] // Batch_Size
   
        for epoch in range(MAX_EPOCH):

            if inter_kd_mode == 'DCA':
                if epoch % dca_learn_freq == 0:  
                    print('Learning Student DCA ...')              
                    student_DCA_weights = Learn_DCA(images_pl, Xte, student_inter_vec, y_dca, sess, student_DCA, updated_mask_dict)
        
            shuf_ind = list(range(Xtr.shape[0]))
            np.random.shuffle(shuf_ind)
            shuf_Xtr,shuf_ytr = Xtr[shuf_ind,:],ytr[shuf_ind]
            training_loss_list = []
            
            epoch_start = time.time()
            epoch_best_acc = 0
            current_lr = init_lr * (lr_drop_rate ** (epoch // lr_drop_epoch))
            
            best_acc, epoch_best_acc, saved_network = EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network, updated_mask_dict)
        
            # --------------------------------------- Training Step ---------------------------------------
            for ind in range(iter_per_epoch):
                mb_x, mb_y = Get_Batch(ind, iter_per_epoch, shuf_Xtr, shuf_ytr)
                if inter_kd_mode == 'Hint':
                    teacher_inter, teacher_label = Get_Teacher_Outputs(teacher_images_pl, teacher_target, teacher_inter_tensor, mb_x, sess)
                    key_val_list = [(images_pl, mb_x), (labels_pl, mb_y), (lr_pl, current_lr), (soft_target_pl, teacher_label), (teacher_inter_tensor_pl, teacher_inter)]
                elif inter_kd_mode == 'DCA':
                    teacher_dca_trans, teacher_label = sess.run([teacher_trans, teacher_target], {teacher_images_pl: mb_x, teacher_dca_pl: teacher_DCA_weights})
                    key_val_list = [(images_pl, mb_x), (labels_pl, mb_y), (lr_pl, current_lr), (soft_target_pl, teacher_label), (teacher_trans_pl, teacher_dca_trans), (student_dca_pl, student_DCA_weights)]

                training_loss_val = TrainStep(key_val_list, updated_mask_dict, train_op, sess, entropy_loss, ridge_loss, output_kd_loss, inter_kd_loss)
                training_loss_list.append(training_loss_val)
        
                # --------------------------------------- Evaluation Step ---------------------------------------
                if (epoch >= 150) and (ind % 30 == 29):
                    best_acc, epoch_best_acc, saved_network = EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network, updated_mask_dict)    
        
            # Epoch Status Printing
            epoch_end = time.time()
            entropy_loss_val, ridge_loss_val, output_kd_loss_val, inter_kd_loss_val = np.mean(training_loss_list, axis = 0)
            print('Epoch #' + str(epoch) + ' Takes time: ' + str(round(epoch_end - epoch_start,2)) +
                  ' CE Loss: ' + str(round(entropy_loss_val, 4)) + ' Ridge Loss: '  +  str(round(ridge_loss_val, 4)) + ' Out-KD Loss: ' + str(round(output_kd_loss_val, 4)) + ' Int-KD Loss: ' + str(round(inter_kd_loss_val, 4)) + 
                  ' Epoch Best Acc: ' + str(epoch_best_acc) +  ' Best Acc: ' + str(best_acc) + ' Current Lr: ' + str(current_lr)
                 )
   
        end_time = time.time()
        print('\n' + 'Total Time For Training: ' + str(end_time - start_time))
            
        log_stats = {'testing acc': te_acc_list, 'params':train_params}
        
        # Validate that the teacher model is not updated
        teacher_pred = sess.run(teacher_target, {teacher_images_pl: Xte})
        teacher_label = np.argmax(teacher_pred, axis = 1)
        teacher_acc = sum([int(teacher_label[i]) == int(yte[i]) for i in range(len(yte))])/len(yte)
        print('Teacher Accuracy: ' + str(teacher_acc))

        FinalNetworkEvaluation(mResNet, saved_network, mask_list, Xte, yte, log_stats, sess)

if __name__ == '__main__':
    main()    
