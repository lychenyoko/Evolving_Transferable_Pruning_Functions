import numpy as np
import tensorflow as tf
from tensorflow import keras

import time

from Net_Structure.ResNet import ResNet
from Util.training_util import augment_all_images, LogTraining
from Util.Calculators import ResNetParamCal, ResNetFLOPCal, Print_Network

'''-------Load Hyper-parameters and Set Them to Global Parameters------'''
from ResNet_hyperparams_training import *

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--lr_drop_epoch',type = int)
parser.add_argument('--lr_drop_rate',type = float)
parser.add_argument('--ridge',type = float)
parser.add_argument('--model',type = str)
parser.add_argument('--init_lr', type = float)
parser.add_argument('--opt', type = str)
parser.add_argument('--bn_momentum', type = float)
parser.add_argument('--n_res_block', type = int)
parser.add_argument('--load_on_lr_drop', dest='load_on_lr_drop', action='store_true')
parser.add_argument('--no-load_on_lr_drop', dest='load_on_lr_drop', action='store_false')
parser.set_defaults(load_on_lr_drop=False) # whether to load the best model on the lr drop or not
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
if args.n_res_block is not None:
    n_res_block = args.n_res_block
load_on_lr_drop = args.load_on_lr_drop

if block_type == 'Normal':
    num_layer = n_res_block * 6 + 2
if block_type == 'BottleNeck':
    num_layer = n_res_block * 9 + 2

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
    if (model_loaded) and (model != 'None'):
        model_value = np.load(model, allow_pickle = True)
        mResNet = ResNet(param_list = list(model_value), block_type = block_type, num_res_blocks = n_res_block, BN_Var_Epsilon = bn_var_epsilon, BN_Momentum = bn_momentum)
    else:
        mResNet = ResNet(block_type = block_type, num_res_blocks = n_res_block, BN_Var_Epsilon = bn_var_epsilon, BN_Momentum = bn_momentum, Dataset = Dataset)

    return mResNet


def GetTrainOp(mResNet):
    '''-------------- Training Ops --------------'''
    # Generate placeholders for the images and labels.    
    images_pl = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS,IMAGE_PIXELS,IMAGE_CHANNEL))
    labels_pl = tf.placeholder(tf.int64, shape=(None))

    # Build a Graph that computes predictions from the inference model.
    layers, logits = mResNet.inference(images_pl, is_training = True)
    
    # Add to the Graph the Ops for loss calculation.
    entropy_loss, ridge_loss = mResNet.loss(logits, labels_pl, m_ridge)
    training_loss = entropy_loss + ridge_loss
    
    # Add to the Graph the Ops that calculate and apply gradients.
    lr_pl = tf.placeholder(tf.float32)
    train_op = mResNet.training(opt, training_loss, lr_pl, momentum = opt_momentum)
     
    return images_pl, labels_pl, lr_pl, layers, train_op


def GetTestScore(mResNet, Xte, yte):
    dummy, test_logits = mResNet.inference(Xte.astype(np.float32), is_training = False)
    correct_prediction = tf.equal(tf.argmax(test_logits, 1), yte.reshape(-1))
    acc_score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
    return acc_score   


def PrintExperimentStatus():
    '''
    Print the hyper-parameters for training and pruning
    '''
    
    print('\n' + '----------- Training Start -----------' + '\n')
    print('Training Params: ' + '\n' +  '\n' + 

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
         '    LR Drop Rate: ' + str(lr_drop_rate) + '\n' + 
         '    Load on LR Drop: ' + str(load_on_lr_drop) + '\n' + '\n' + 

         '  Regularization Scheme: ' + '\n' +    
         '    Ridge: ' + str(m_ridge) + '\n' +       
         '    Data Augmentation: ' + str(use_data_aug) + '\n' +
         '    BN Momentum: ' + str(bn_momentum) + '\n' +
         '    BN Var Epsilon: ' + str(bn_var_epsilon) + '\n'        
    )
    
    if model_loaded:
        print('Init Model:' + '\n'
              '  ' + str(model) + '\n'
    )       


def EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network):
    # --------------------------------------- Evaluation Step ---------------------------------------
    te_acc = sess.run(acc_score)
    te_acc_list.append(te_acc)
    
    if te_acc >= best_acc:
        best_acc = te_acc
        saved_network = mResNet.model_save(sess)
    
    if te_acc >= epoch_best_acc:
        epoch_best_acc = te_acc
    
    return best_acc, epoch_best_acc, saved_network

def TrainStep(ind, iter_per_epoch, shuf_Xtr, shuf_ytr, current_lr, images_pl, labels_pl, lr_pl, train_op, sess):
    if (ind != (iter_per_epoch - 1)):
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:(ind+1)*Batch_Size,:],shuf_ytr[ind*Batch_Size:(ind+1)*Batch_Size,:]
    else:
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:,:],shuf_ytr[ind*Batch_Size:,:]
    if use_data_aug:
        mb_x = augment_all_images(mb_x,4)

    train_dict = dict()
    train_dict[images_pl] = mb_x
    train_dict[labels_pl] = mb_y
    train_dict[lr_pl] = current_lr

    _ = sess.run(train_op,train_dict)

def FinalNetworkEvaluation(mResNet, saved_network, log_stats, best_acc):

    final_acc =  best_acc
    final_FLOP = ResNetFLOPCal(mResNet)
    final_param = ResNetParamCal(mResNet)
     
    LogTraining(saved_network = saved_network, log_stats = log_stats,
                best_acc = final_acc, FLOP = final_FLOP, param = final_param, num_layer = num_layer)

def main():
    '''
    Main method of the script
    '''
    Xtr, Xte, ytr, yte = LoadCIFAR(Dataset) # Dataset Loading

    '''-------Training Graph and Operation------'''
    mResNet = GetTrainGraph()
    images_pl, labels_pl, lr_pl, layers, train_op = GetTrainOp(mResNet)
        
    '''-------Testing Acc Tensor -------'''
    acc_score = GetTestScore(mResNet, Xte, yte)
  
    # Initialize the Training Grpah
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Start time counter 
    start_time = time.time()

    ##--------------------------------- Experiment Begins ----------------------------------
    PrintExperimentStatus()
    train_params = {'Dataset': Dataset, 'Epochs': MAX_EPOCH, 'Batch Size':Batch_Size, 
                    'Init LR':init_lr,  'LR Drop Epoch':lr_drop_epoch, 'LR Drop Rate': lr_drop_rate, 
                    'Optimizer':opt, 'Opt Momentum':opt_momentum,
                    'Ridge': m_ridge, 'Data Aug': use_data_aug, 'BN Momentum': bn_momentum, 'BN Var Epsilon': bn_var_epsilon}

    # >>> Training Process
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
        if (load_on_lr_drop) and (epoch % lr_drop_epoch == 0) and (epoch > 0):
            mResNet.model_load(saved_network, sess)
        
        best_acc, epoch_best_acc, saved_network = EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network)
    
        # --------------------------------------- Training Step ---------------------------------------
        for ind in range(iter_per_epoch):
            TrainStep(ind, iter_per_epoch, shuf_Xtr, shuf_ytr, current_lr, images_pl, labels_pl, lr_pl, train_op, sess)
    
            # --------------------------------------- Evaluation Step ---------------------------------------
            if (epoch >= 150) and (ind % 30 == 29):
                best_acc, epoch_best_acc, saved_network = EvaluationStep(mResNet, sess, acc_score, te_acc_list, best_acc, epoch_best_acc, saved_network)    
    
        # Epoch Status Printing
        epoch_end = time.time()
        print('Epoch #' + str(epoch) + ' Takes time: ' + str(round(epoch_end - epoch_start,2)) + ' Epoch Best Acc: ' + str(epoch_best_acc) +
            ' Best Acc: ' + str(best_acc) + ' Current Lr: ' + str(current_lr))
   
    end_time = time.time()
    print('\n' + 'Total Time For Training: ' + str(end_time - start_time))
        
    log_stats = {'testing acc': te_acc_list, 'params':train_params}
    FinalNetworkEvaluation(mResNet, saved_network, log_stats, best_acc)

if __name__ == '__main__':
    main()    
