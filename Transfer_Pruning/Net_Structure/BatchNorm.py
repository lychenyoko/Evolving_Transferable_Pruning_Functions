# ------------------------------------------------------------------------------------------------------------
# This Code Implements Batch Normalization in TensorFlow
# Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
# ------------------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# Reference CONSTANTS
# BN_VAR_EPSILON = 1e-5, BN_MOMEMTUM = 0.99

class BatchNormLayerForCNN:
    
    def __init__(self, num_channel=0, BN_param_list = None, Var_Epsilon = 1e-5, Momentum = 0.99):

        '''
        Usage:
            Initialization of the BN object
        Args:
            num_channel:    The number of channel that will pass through this BN object
            BN_param_list:  If not None, the 4 parameter of the BN will be initialized with it
            Var_Epsilon:    The episilon parameter for BN varaince
            Momentum:       The momentum to update the BN inference mean and inference var
        Returns:
            A BN object
        '''

        # Initialize the BN training parameters
        if BN_param_list == None:
            self.scale = tf.Variable(tf.ones(num_channel))            
            self.offset = tf.Variable(tf.zeros(num_channel))
            self.inference_mean = tf.Variable(tf.zeros(num_channel),trainable = False)
            self.inference_var = tf.Variable(tf.ones(num_channel), trainable = False)
        else:
            self.scale = tf.Variable(BN_param_list[0],name = 'BN_scale')    
            self.offset = tf.Variable(BN_param_list[1], name = 'BN_offset')            
            self.inference_mean = tf.Variable(BN_param_list[2], trainable = False, name = 'BN_inference_mean')
            self.inference_var = tf.Variable(BN_param_list[3], trainable = False, name = 'BN_inference_var')

        # The hyperparameters for BN
        self.var_epsilon = Var_Epsilon
        self.momentum = Momentum
        
        # param list of the BN object
        self.param_list = [self.scale,self.offset,self.inference_mean,self.inference_var]

        # Two lists of BN statistics for multi-gpu training where the samples are distributed on different GPUs
        self.batch_mean_list = []
        self.batch_square_mean_list = []


    def feed_forward(self,fea_map_input,is_training):

        '''
        Usage:
            Inference path of a BN object
        Args:
            fea_map_input: A tensor of shape [N,H,W,C] where C matches the size of BN parameters
            is_training: Whether the BN is used in training phase or inference phase
       
        Returns:
            The output of the BN
        '''

        if is_training == True:
            # Flatten 4D tensor to 2D for batch_mean and batch_var calculation
            flated_inp = tf.reshape(fea_map_input,[-1,fea_map_input.shape[-1]])
            
            # Multiple GPU strategy, use the batch_mean_list and bath_square_mean_list for inf_mean and inf_var update 
            batch_mean = tf.reduce_mean(flated_inp,axis=0,name='batch_mean')
            batch_square_mean = tf.reduce_mean(tf.square(flated_inp),axis=0,name='batch_square_mean')
            batch_var = batch_square_mean - tf.square(batch_mean)
            self.batch_mean_list.append(batch_mean)
            self.batch_square_mean_list.append(batch_square_mean)

            # Obtain output by tf.nn.bn method        
            fea_map_output = tf.nn.batch_normalization(fea_map_input, batch_mean, batch_var, self.offset, self.scale, self.var_epsilon)
            
            return fea_map_output
        else:
            fea_map_output = tf.nn.batch_normalization(
                fea_map_input, self.inference_mean, self.inference_var, self.offset, self.scale, self.var_epsilon
            )
            return fea_map_output
            


    def update_moving_mean_var(self):

        '''
        Usage:
            Get the avg_batch_mean and avg_batch_var for inf_mean and inf_var update
       
        Returns:
            The operation for inf_mean and inf_var update
        '''

        # Get avg_batch_mean and avg_batch_square_mean
        avg_batch_mean = tf.reduce_mean(self.batch_mean_list, axis=0)
        avg_batch_var = tf.reduce_mean(self.batch_square_mean_list, axis=0) - tf.square(avg_batch_mean)
        self.avg_batch_mean = avg_batch_mean
        self.avg_batch_var = avg_batch_var
        
        # Inf_mean and inf_var update operation
        self.moving_mean = tf.assign(self.inference_mean, self.inference_mean * self.momentum + (1 - self.momentum) * avg_batch_mean, name = 'BN_inf_mean_update')
        self.moving_var = tf.assign(self.inference_var, self.inference_var * self.momentum + (1 - self.momentum) * avg_batch_var, name = 'BN_inf_var_update')
        return [self.moving_mean,self.moving_var]

    def var_initialization(self):
        '''
        Usage:
            Locally initialized the BN Object
        Returns:
            List of BN variable initializer
        '''
        
        bn_var_init_list = []
        for var in self.param_list:
            bn_var_init_list.append(var.initializer)
        return bn_var_init_list

    def bn_load(self,param_list):
        '''
        Usage:
            Load parameters from a list to modify BN param
        Args:
            param_list: List of parameter to load into the BN object
        Returns:
            List of BN parameter assignment operation
        '''

        bn_assign_op = []
        for self_param, param in zip(self.param_list,param_list):
            bn_assign_op.append(tf.assign(self_param,param))
        return bn_assign_op
