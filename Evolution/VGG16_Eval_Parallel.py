import numpy as np
import tensorflow as tf
from tensorflow import keras

# Python Built-in Package
import time
import random
import pickle as pkl
import argparse

# Evolution Package
from deap import gp 

# Self Implemented Functions
from Net_Structure import VGG16
from GA_Helper.Evaluation_Util import Prune_Nodes_Channel_By_Func, augment_all_images, VGG16_Model_Save
from GA_Helper.VGG16_Calculators import VGG16_FLOPCal, VGG16_ParamCal
from GA_Helper.newPrimSet import DFunc_Individual, Single_Tree_Pset, arg_for_DF, Metric_Score

'''-------HyperParams------'''
from VGG16_Eval_Hyperparams import *

# --------------------------------- Evaluation Parameters ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--indv',type=str)
parser.add_argument('--gen',type=str)
parser.add_argument('--i',type=str)
parser.add_argument('--dir',type=str)
parser.add_argument('--init_lr', type=float)
parser.add_argument('--lr_drop', type=float)
parser.add_argument('--k_prob', type=float)
parser.add_argument('--opt', type = str)
args = parser.parse_args()

# For optimizing hyper-params
if args.init_lr is not None:
    init_lr = args.init_lr
if args.lr_drop is not None:
    lr_drop_rate = args.lr_drop
if args.k_prob is not None:
    dropout_keep_prob = args.k_prob
if args.opt is not None:
    opt = args.opt


# The Individual
individual = pkl.load(open(args.indv,'rb'))
ind_func = gp.compile(expr = individual.expr_tree, pset = individual.pset)
pruned_func = lambda X,y,WI,W: Metric_Score(arg = arg_for_DF(X,y) + [WI, W], func = ind_func)

# The Logging Params
genid = args.gen
indid = args.i
saving_dir = args.dir


def Load_CIFAR10_Data():
    '''
    Usage:
        Load the CIFAR10 dataset
    '''
    CIFAR10 = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = CIFAR10.load_data()
    mean = 120.707
    std = 64.15
    Xtr = (train_images - mean)/(std + 1e-7)
    Xte = (test_images - mean)/(std + 1e-7) # The way for data standardization a bit different from MNIST
    ytr = train_labels
    yte = test_labels

    return Xtr, Xte, ytr, yte


def Build_Train_Graph():
    '''-------Training Graph------'''
    # Tell TensorFlow that the model will be built into the default Graph.
    tf.reset_default_graph()    
    if model_loaded:
        mVGG16 = VGG16.VGG16(param_list = np.load(m_model, allow_pickle = True))
    else:
        mVGG16 = VGG16.VGG16()
    return mVGG16


def Get_Train_Op(mVGG16):
    # Generate placeholders for the images and labels and array mask
    images_pl = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS,IMAGE_PIXELS,IMAGE_CHANNEL))
    labels_pl = tf.placeholder(tf.int64, shape=(None))
    mask_pl_list = mVGG16.get_mask_pl_list() # The TF mask place holder!
    
    '''-------------- Training Ops --------------'''
    # Build a Graph that computes predictions from the inference model.
    k_prob_pl = tf.placeholder(tf.float32)
    dummy, logits = mVGG16.inference(images_pl,True,k_prob_pl,mask_pl_list) # Inference for Training
    tf_dict, dum  = mVGG16.inference(images_pl,False,1,mask_pl_list) # Inference for Pruning
    
    # Add to the Graph the Ops for loss calculation.
    entropy_loss,ridge_loss = mVGG16.loss(logits, labels_pl, m_ridge)
    training_loss = entropy_loss + ridge_loss
    
    # Add to the Graph the Ops that calculate and apply gradients.
    lr_pl = tf.placeholder(tf.float32)
    train_op = mVGG16.training(opt, training_loss, lr_pl)

    return images_pl, labels_pl, mask_pl_list, k_prob_pl, lr_pl, tf_dict, train_op


def Get_Test_Op(mVGG16, Xte, yte, mask_pl_list):

    '''-------------- Testing Ops --------------'''
    dummy, test_logits = mVGG16.inference(Xte.astype(np.float32),False,1,mask_pl_list)
    correct_prediction = tf.equal(tf.argmax(test_logits, 1), yte.reshape(-1))
    acc_score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
    
    return acc_score


def Print_Experiment_Information(mVGG16):
    #####--------------------------------- Training Begin ----------------------------------
    print('\n' + '----------- Training Start -----------')
    print('Evaluating ' + str(indid) + ' ind in ' + str(genid) +' generation.')
    print(individual.expr_tree)  
    
    print('\n' + 'Training Params: ' + '\n' + 
         '  Epochs: ' + str(MAX_EPOCH) + '\n' +
         '  Batch Size: ' + str(Batch_Size) + '\n' +          
         
         '  Ridge: ' + str(m_ridge) + '\n' + 
         '  Data Augmentation: ' + str(use_data_aug) + '\n' +
         '  Dropout: ' + str(dropout_keep_prob) + '\n' + 
         
         '  Optimizer: ' + opt + '\n' +
         '  Momentum: ' + str(MOMENTUM) + '\n' +          
         '  Init LR: ' + str(init_lr) + '\n' +
         '  LR Drop Rate: ' + str(lr_drop_rate) + '\n' + 
         '  LR Drop Epoch1: ' + str(lr_drop_epoch1) + '\n' + 
         '  LR Drop Epoch2: ' + str(lr_drop_epoch2) + '\n' + '\n' +
    
         'Pruning Params: ' + '\n' + 
         '  Prune Step: ' + str(Prune_Step) + '\n'       
    )    
    prune_list = ['  Conv' + str(i+1) + '_Sel_Remv: ' + str((sel_list[i],rmve_list[i])) for i in range(13)]
    for prune in prune_list:
        print(prune)
    print('  Fc1_Sel_Rmve: ' +  str((sel_list[-1],rmve_list[-1])) + '\n')
    
    print('Init Model: ' + str(m_model))
    print('Initial Model Shape:')
    mVGG16.print_shape()

def Print_Mask_List(mask_list):
    mask_shape = [sum(mask) for mask in mask_list]
    print('Current mask shape is: ' + str(mask_shape) + '\n')


def Evaluation_Step(mVGG16, mask_pl_list, mask_list, sess, acc_score, epoch_best_acc, best_acc, best_network):

    # ---------------------------- Testing and saving network state ----------------------------
    test_feed_dict = {}
    for pl,val in zip(mask_pl_list, mask_list):
        test_feed_dict[pl] = val
    te_acc = sess.run(acc_score,feed_dict = test_feed_dict)
    
    if te_acc >= epoch_best_acc:
        epoch_best_acc = te_acc
    
    if te_acc >= best_acc:
        best_acc = te_acc    
        best_network = mVGG16.model_save(sess)
    return epoch_best_acc, best_acc, best_network


def Train_Step(ind, iter_per_epoch, Batch_Size, shuf_Xtr, shuf_ytr, images_pl, labels_pl, lr_pl, k_prob_pl, current_lr, train_op, sess, mask_pl_list, mask_list):
    '''
    Usage:
        One update step of gradient descent
    '''

    if (ind != (iter_per_epoch - 1)):
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:(ind+1)*Batch_Size,:],shuf_ytr[ind*Batch_Size:(ind+1)*Batch_Size,:]
    else:
        mb_x,mb_y = shuf_Xtr[ind*Batch_Size:,:],shuf_ytr[ind*Batch_Size:,:]
    if use_data_aug:
        mb_x = augment_all_images(mb_x,4)
    
    train_feed_dict = {images_pl:mb_x, labels_pl:mb_y, lr_pl: current_lr, k_prob_pl: dropout_keep_prob}
    for pl,val in zip(mask_pl_list,mask_list):
        train_feed_dict[pl] = val
    _ = sess.run(train_op,feed_dict = train_feed_dict)

def Final_Prune_Net_Evaluation(mVGG16, final_network, final_mask_list, Xte, yte, sess):
    '''
    Usage:  
        Assess and save the final pruned network in the end
    '''
    m_compressed_net = mVGG16.compressed_model(final_network, final_mask_list, )
    comp_VGG = VGG16.VGG16(m_compressed_net)
    dummy, te_logits = comp_VGG.inference(Xte.astype(np.float32),False,1)
    corr_prediction = tf.equal(tf.argmax(te_logits, 1), yte.reshape(-1))
    acc_ = tf.reduce_mean(tf.cast(corr_prediction, tf.float32), name = 'final_acc')
    comp_VGG.var_initialization(sess)
    final_acc = sess.run(acc_)

    FLOP_retain = VGG16_FLOPCal(comp_VGG)
    Param_retain = VGG16_ParamCal(comp_VGG) 
    
    # ------------------ Final Model Info Printing ------------------
    print('Final Model Status:')
    comp_VGG.print_shape()    
    print('Acc: ' + str(round(final_acc * 100, 2)) + '\n' +
          'FLOP retain: ' + str(round(FLOP_retain * 100, 2)) + '\n' +
          'Param retain: ' + str(round(Param_retain * 100, 2))
    )
    
    # ------ Logs and Model Saving Part ------
    VGG16_Model_Save(m_compressed_net, final_acc, FLOP_retain, Param_retain, genid, indid, saving_dir)


def main():
    Xtr, Xte, ytr, yte = Load_CIFAR10_Data()
    mVGG16 = Build_Train_Graph()
    images_pl, labels_pl, mask_pl_list, k_prob_pl, lr_pl, tf_dict, train_op = Get_Train_Op(mVGG16)
    acc_score = Get_Test_Op(mVGG16, Xte, yte, mask_pl_list)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    # Print the experiment status
    Print_Experiment_Information(mVGG16)

    # Start time counter 
    start_time = time.time()

    # Define number of iteration in each epoch
    iter_per_epoch = Xtr.shape[0]//Batch_Size

    # Control Signal for Pruning Process
    mask_list = [np.array([True] * mask.shape[0]) for mask in mask_pl_list] # Defined it to be a list of list
    best_network = mVGG16.model_save(sess)

    #####--------------------------------- Actual Computation Happens ----------------------------------
    for step in range(Prune_Step):
        
        print('\n' + 'Pruning Step: ' + str(step + 1))                
        Print_Mask_List(mask_list)

        # --------------------------- Pruning Logic ---------------------------    
        mVGG16.model_recovery(best_network, sess) # Recovery the network to previous best network
        mask_dict = {}
        for pl,val in zip(mask_pl_list,mask_list):
            mask_dict[pl] = val
    
        cur_acc = sess.run(acc_score,feed_dict = mask_dict)
        print('The current accuracy is: ' + str(cur_acc))

        W_list = sess.run(mVGG16.network_param_list)[::2][:-1]
    
        # Iteratively Evaluate the Layer of Feature Maps and prune them out
        for layer_i in range(len(sel_list)):
    
            if rmve_list[layer_i] == 0:
                continue
    
            layer_name,layer_tensor = list(tf_dict.items())[layer_i]
            lay_shape = list(layer_tensor.shape)
            lay_shape[0] = Xtr.shape[0]
            lay_out = np.zeros(lay_shape)
            for i in range(5):
                prune_dict = dict(mask_dict)
                prune_dict[images_pl] = Xtr[10000*i:10000*(i+1)]
                tmp_out = sess.run(layer_tensor, feed_dict=prune_dict)
                lay_out[10000*i:10000*(i+1)] = tmp_out
    
            layer_mask,layer_sel,layer_rmv = mask_list[layer_i],sel_list[layer_i],rmve_list[layer_i]
           
            # Pruning maps/nodes
            if (sum(layer_mask) > layer_rmv):
                picked_node = Prune_Nodes_Channel_By_Func(
                        lay_out[...,layer_mask], ytr, pruned_func, layer_sel, layer_rmv, W_list[layer_i][..., layer_mask]
                    )
                mask_ind = np.where(layer_mask == True)[0][picked_node]
                layer_mask[mask_ind] = False
                print('We have masked out  #' + str(mask_ind) + ' in ' + layer_name + '. It will have ' 
                    + str(sum(layer_mask)) +' nodes/maps.')
            del lay_out # Deallocate for the memory
            
        print('')
        Print_Mask_List(mask_list)

        # --------------------------- Training Logic ---------------------------
        best_acc = 0
        current_lr = init_lr

        for epoch in range(MAX_EPOCH):
            
            epoch_start = time.time()

            # Initial Evaluation
            epoch_best_acc = 0
            epoch_best_acc, best_acc, best_network = Evaluation_Step(mVGG16, mask_pl_list, mask_list, sess, acc_score, epoch_best_acc, best_acc, best_network)

            # ---------------------- The Training Process! ----------------------
            if (epoch == lr_drop_epoch1) or (epoch == lr_drop_epoch2):
                mVGG16.model_recovery(best_network,sess)
                current_lr = current_lr * lr_drop_rate
                            
            shuf_ind = list(range(Xtr.shape[0]))
            np.random.shuffle(shuf_ind)
            shuf_Xtr,shuf_ytr = Xtr[shuf_ind,:],ytr[shuf_ind]
        
            for ind in range(iter_per_epoch):        
                Train_Step(ind, iter_per_epoch, Batch_Size, shuf_Xtr, shuf_ytr, images_pl, labels_pl, lr_pl, k_prob_pl, current_lr, train_op, sess, mask_pl_list, mask_list)       
        
                # ---------------------------- Testing and saving network state ----------------------------
                if ind % 30 == 29:                    
                    epoch_best_acc, best_acc, best_network = Evaluation_Step(mVGG16, mask_pl_list, mask_list, sess, acc_score, epoch_best_acc, best_acc, best_network)
                # ---------------------------- Testing and saving network state ----------------------------
                                  
            epoch_end = time.time()
            print('Epoch #' + str(epoch) + ' Takes time: ' + str(round(epoch_end - epoch_start,2)) + ' Current Lr: ' + str(current_lr) 
                +  ' Epoch Best Acc: ' + str(epoch_best_acc) + ' Best Acc: ' + str(best_acc))
        
        end_time = time.time()

    print('The whole training takes: ' + str(round(end_time - start_time, 2)))
    final_network = best_network
    final_mask_list = mask_list
    Final_Prune_Net_Evaluation(mVGG16, final_network, final_mask_list, Xte, yte, sess)

if __name__ == '__main__':
    main()
