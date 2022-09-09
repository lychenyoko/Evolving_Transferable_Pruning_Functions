import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
import pickle as pkl

import tensorflow as tf
from tensorflow import keras

from deap import gp 

from Net_Structure import LeNet5
from GA_Helper.Evaluation_Util import Prune_Nodes_Channel_By_Func, LeNet5_Model_Save, mask_fla_from_conv2
from GA_Helper.LeNet5_Calculators import LeNet5_FLOPCal, LeNet5_ParamCal
from GA_Helper.newPrimSet import DFunc_Individual, Single_Tree_Pset, arg_for_DF, Metric_Score

# --------------------------------- Training Hyper-parameters ---------------------------------
from LeNet5_Eval_Hyperparams import *

step_param_list = [conv1_sel_list,conv1_rmv_list,conv2_sel_list,
		    conv2_rmv_list,fla_sel_list,fla_rmv_list,a1_sel_list,a1_rmv_list]

# --------------------------------- Evaluation Parameters ---------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indv',type=str)
parser.add_argument('--gen',type=str)
parser.add_argument('--i',type=str)
parser.add_argument('--dir',type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--ridge', type=float)
args = parser.parse_args()

# For hyper-paramter optimization
if args.lr is not None:
    learning_rate = args.lr
if args.ridge is not None:
    m_ridge = args.ridge

# The Individual
individual = pkl.load(open(args.indv,'rb'))
ind_func = gp.compile(expr = individual.expr_tree, pset = individual.pset)
pruned_func = lambda X,y,WI,W: Metric_Score(arg = arg_for_DF(X,y) + [WI, W], func = ind_func)

# The Logging Params
genid = args.gen
indid = args.i
saving_dir = args.dir

def Load_MNIST_data():
    '''
    Usage:
        Load MNIST data
    '''
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    Xtr = train_images.reshape(train_images.shape[0],-1)/255
    Xte = test_images.reshape(test_images.shape[0],-1)/255
    ytr = train_labels
    yte = test_labels
    return Xtr, ytr, Xte, yte

def Build_Train_Graph():
    '''
    Usage:
        Build LeNet-5 Graph
    '''

    tf.reset_default_graph()
    if model_loaded:
        model,fla_sel = np.load(model_name, allow_pickle = True)
        mLeNet = LeNet5.LeNet_5(model,fla_sel)
    else:
        mLeNet = LeNet5.LeNet_5()

    fla_true_ind = np.where(np.array(mLeNet.fla_sel) == True)[0]
    return mLeNet, fla_true_ind


def Get_Train_Op(mLeNet):

    # Generate placeholders for the images and labels.
    images_pl = tf.placeholder(tf.float32, shape=(None,LeNet5.IMAGE_PIXELS))
    labels_pl = tf.placeholder(tf.int64, shape=(None))
    conv1_pl,conv2_pl,flat_pl,fc1_pl = mLeNet.get_mask_pl()
    
    # Build a Graph that computes predictions from the inference model.
    tf_dict, logits = mLeNet.inference(images_pl,conv1_pl,conv2_pl,flat_pl,fc1_pl)
    
    # Add to the Graph the Ops for loss calculation.
    entropy_loss = mLeNet.loss(logits, labels_pl, m_ridge)
    
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mLeNet.training(opt, entropy_loss, learning_rate)

    return images_pl, labels_pl, conv1_pl, conv2_pl, flat_pl, fc1_pl, tf_dict, train_op


def Get_Test_Op(conv1_pl,conv2_pl,flat_pl,fc1_pl,Xte,yte,mLeNet):
    # explicitly testing operation
    dummy, test_logits = mLeNet.inference(Xte.astype(np.float32),conv1_pl,conv2_pl,flat_pl,fc1_pl)
    correct_prediction = tf.equal(tf.argmax(test_logits, 1), yte)
    acc_score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
    return acc_score

def Print_Experiment_Information(mLeNet):

    print('\n'+'----------------- Training Start -----------------')
    print('Evaluating ' + str(indid) + ' ind in ' + str(genid) +' generation.')
    print(individual.expr_tree)    
    
    print('\n' + 'Training Params: ' + '\n' +     
         '  Epochs: ' + str(MAX_EPOCH) +'\n' +
         '  Learning Rate: ' + str(learning_rate) + '\n' +
         '  Optimizer: ' + opt + '\n' +    
         '  Ridge: ' + str(m_ridge) + '\n' + 
         '  Model loaded: ' + str(model_loaded) +'\n' + '\n' +
         
         'Pruning Params: ' + '\n' +          
         '  Pruning Stage: ' + str(Prune_Step) + '\n' + 
         '  conv1 sel/rmvd: ' + str(list(zip(conv1_sel_list,conv1_rmv_list))) + '\n' +  
         '  conv2 sel/rmvd: ' + str(list(zip(conv2_sel_list,conv2_rmv_list))) + '\n' +
         '  fla sel/rmvd: ' + str(list(zip(fla_sel_list,fla_rmv_list))) + '\n'      
         '  a1 sel/rmvd: ' + str(list(zip(a1_sel_list,a1_rmv_list))) + '\n' 
    )
    
    print('The Initial Model Shape is:')
    print(Get_Network_Shape(mLeNet))
    print('')

def Get_Network_Shape(mLeNet):
    return [int(mLeNet.b_conv1.shape[0]), int(mLeNet.b_conv2.shape[0]), int(mLeNet.W_fc1.shape[0]), int(mLeNet.b_fc1.shape[0])]


def Get_Default_Mask(mask_pl_list):
    mask_conv1, mask_conv2, mask_fla, mask_a1 = [np.array([True] * pl.shape[0]) for pl in mask_pl_list]
    return mask_conv1, mask_conv2, mask_fla, mask_a1

def Print_Mask(mask_conv1, mask_conv2, mask_fla, mask_a1):
    print('\n' + 'Current network mask indicates:' + '\n' +
          'conv1 has # of maps: ' + str(sum(mask_conv1)) + '\n' +
          'conv2 has # of maps: ' + str(sum(mask_conv2)) + '\n' +
          'fla has # of nodes: ' + str(sum(mask_fla)) + '\n' +
          'a1 has # of nodes: ' + str(sum(mask_a1))
    )           


def Prune_A_Layer(lay_mask, lay_rmv, lay_sel, lay_out, lay_name, ytr, ker):
    '''
    Usage:
        To update the mask of a layer based on the scores of the channels

    Args:
        lay_mask: (list) of bool of the number of nodes/channels in the layer
        lay_rmv: (int) of number to removed nodes from the layer
        lay_sel: (int) of number of selected nodes from the layer
        lay_out: (np.array) of the feature maps or the nodes in representing the layer
        ytr: (np.array) of labels
        ker: (np.array) of the incoming kernel of the layer
    '''
    if (sum(lay_mask) > lay_rmv) and (lay_rmv > 0):
        picked_node = Prune_Nodes_Channel_By_Func(
            lay_out[..., lay_mask], ytr, pruned_func, lay_sel, lay_rmv, ker[..., lay_mask]
        )
        mask_ind = np.where(lay_mask == True)[0][picked_node]
        lay_mask[mask_ind] = False
        print('We have masked out #: ' + str(mask_ind) + ' in ' + lay_name)

def Evaluation_Step(best_acc, epoch_best_acc, best_network, mask_dict, sess, acc_score, mLeNet):
     '''
     Usage:
         Evaluate the accuracy of current state and update the best state conditions
     '''
     te_acc = sess.run(acc_score, feed_dict = mask_dict)
    
     if te_acc >= epoch_best_acc:
         epoch_best_acc = te_acc
    
     if te_acc >= best_acc:
         best_acc = te_acc    
         best_network = mLeNet.model_save(sess)

     return best_acc, epoch_best_acc, best_network

def Train_Step(Xtr,ytr,train_ind,mask_dict,images_pl,labels_pl,train_op,sess):
     '''
     Usage:
         Perform one step of gradient descent
     '''
     mb_x, mb_y = Xtr[train_ind], ytr[train_ind]
     train_dict = dict(mask_dict)
     train_dict[images_pl] = mb_x
     train_dict[labels_pl] = mb_y
     _ = sess.run(train_op, feed_dict= train_dict)



def Final_Prune_Net_Evaluation(sess, mLeNet, final_network, final_mask_list, Xte, yte):
    '''
    Usage:
        The final saving of the pruned model
    '''

    sess.run(mLeNet.model_recovery(final_network))
    pruned_NN = mLeNet.compressed_model(final_mask_list, sess)

    dummy, prune_test_logits = pruned_NN.inference(Xte.astype(np.float32))
    correct_prediction_prime = tf.equal(tf.argmax(prune_test_logits, 1), yte)
    acc_score_prime = tf.reduce_mean(tf.cast(correct_prediction_prime, tf.float32), name = 'accuracy_prime')
    pruned_NN.var_initialization(sess)
    final_acc = sess.run(acc_score_prime)

    prune_shape = Get_Network_Shape(pruned_NN)
    print('The compressed network shape info: ' + str(prune_shape))
    FLOP_retain = LeNet5_FLOPCal(prune_shape)
    Param_retain = LeNet5_ParamCal(prune_shape)
    
    print('Final Network Status: ' + '\n' + 
          'Acc: ' + str(round(final_acc * 100, 2)) + '\n' +
          'FLOPs: ' + str(round(FLOP_retain * 100, 2)) + '\n' +
          'Params: ' + str(round(Param_retain * 100, 2)) )
    
    LeNet5_Model_Save(pruned_NN, sess, final_acc, FLOP_retain, Param_retain, genid, indid, saving_dir)


def main():
   
    # Setting up the dataset, inference graph, training and testing operations
    Xtr, ytr, Xte, yte = Load_MNIST_data()
    mLeNet, fla_true_ind = Build_Train_Graph()
    images_pl, labels_pl, conv1_pl, conv2_pl, flat_pl, fc1_pl, tf_dict, train_op = Get_Train_Op(mLeNet)
    mask_pl_list = [conv1_pl, conv2_pl, flat_pl, fc1_pl]
    acc_score = Get_Test_Op(conv1_pl,conv2_pl,flat_pl,fc1_pl,Xte,yte,mLeNet)

    # Initialization
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Start the Experiments.
    start_time = time.time()
    Print_Experiment_Information(mLeNet)

    mask_conv1, mask_conv2, mask_fla, mask_a1 = Get_Default_Mask(mask_pl_list)
    best_network = mLeNet.model_save(sess)

    for step in range(Prune_Step):
    
        print('\n' + '----------------- Pruning Step ' + str(step) + ' -----------------' + '\n')
    
        # --------------------------------------- Model Pruning ---------------------------------------  
        conv1_sel,conv1_rmv,conv2_sel,conv2_rmv,fla_sel,fla_rmv,a1_sel,a1_rmv \
        = [param[step] for param in step_param_list]
     
        Print_Mask(mask_conv1, mask_conv2, mask_fla, mask_a1)
        mask_dict = {conv1_pl: mask_conv1, conv2_pl: mask_conv2, flat_pl: mask_fla, fc1_pl: mask_a1}

        sess.run(mLeNet.model_recovery(best_network))
        cur_accuracy = sess.run(acc_score, feed_dict=mask_dict)
        print('The current accuracy is: '+ str(cur_accuracy))
            
        Layer_Dict = dict(mask_dict)
        Layer_Dict[images_pl] = Xtr
        conv1_net, conv2_net, fla_net, a1_net = sess.run(
            [ tf_dict['Conv1 Output'], tf_dict['Conv2 Output'], tf_dict['Flatten Output'], tf_dict['Fc1 Output']], 
            feed_dict = Layer_Dict
        )
        W1, W2, W3, W4 = sess.run(mLeNet.var_list)[::2]

        # Pruning conv1
        Prune_A_Layer(lay_mask = mask_conv1, lay_rmv = conv1_rmv, lay_sel = conv1_sel, lay_out = conv1_net, lay_name = 'Conv1', ytr = ytr, ker = W1)

        # Pruning conv2
        Prune_A_Layer(lay_mask = mask_conv2, lay_rmv = conv2_rmv, lay_sel = conv2_sel, lay_out = conv2_net, lay_name = 'Conv2', ytr = ytr, ker = W2)
        mask_fla = mask_fla_from_conv2(mask_conv2, fla_true_ind, mask_fla) # Update mask_fla due to mask_conv2 update

        # Pruning node in flatten layer
        Prune_A_Layer(lay_mask = mask_fla, lay_rmv = fla_rmv, lay_sel = fla_sel, lay_out = fla_net, lay_name = 'Flatten', ytr = ytr, ker = None)        
        # Pruning node in a1 layer
        Prune_A_Layer(lay_mask = mask_a1, lay_rmv = a1_rmv, lay_sel = a1_sel, lay_out = a1_net, lay_name = 'a1', ytr = ytr, ker = W3)  

        Print_Mask(mask_conv1, mask_conv2, mask_fla, mask_a1)
        mask_dict = {conv1_pl: mask_conv1, conv2_pl: mask_conv2, flat_pl: mask_fla, fc1_pl: mask_a1}          
        best_acc = 0        
        
        # --------------------------------------- Model Retraining ---------------------------------------
        for epoch in range(MAX_EPOCH):
    
            kf = StratifiedKFold(n_splits=int(Xtr.shape[0]//batch_size), random_state = epoch + 300)
            epoch_start = time.time()
            epoch_best_acc = 0
    
            best_acc, epoch_best_acc, best_network = Evaluation_Step(best_acc, epoch_best_acc, best_network, mask_dict, sess, acc_score, mLeNet)
    
            for _dummy,train_ind in kf.split(Xtr,ytr):
                Train_Step(Xtr,ytr,train_ind,mask_dict,images_pl,labels_pl,train_op,sess)   
                best_acc, epoch_best_acc, best_network = Evaluation_Step(best_acc, epoch_best_acc, best_network, mask_dict, sess, acc_score, mLeNet)
                   
            epoch_end = time.time()
            print('Epoch #' + str(epoch + 1) + ' Takes time: ' + str(round(epoch_end - epoch_start,2)) + 
                 ' Epoch Best Acc: ' + str(epoch_best_acc) + ' Best Acc: ' + str(best_acc))
                    
    end_time = time.time()
    print('\n' + 'The whole training takes: ' + str(round(end_time - start_time, 2)))
    
    # Evaluate the final pruned model
    final_network = best_network
    final_mask_list = [list(mask) for mask in [mask_conv1,mask_conv2,mask_fla,mask_a1]]
    Final_Prune_Net_Evaluation(sess, mLeNet, final_network, final_mask_list, Xte, yte)

if __name__ == '__main__':
    main()
