# ------------------------------------------------------------------------------------------------------------
# This Code Implements the ResNet Structure Posted On https://github.com/wenxinxu/resnet-in-tensorflow
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# We Adopt the Multibranch Pruning Mechanism Published in He, Yihui, Xiangyu Zhang, and Jian Sun. "Channel pruning for accelerating 
# very deep neural networks." Proceedings of the IEEE International Conference on Computer Vision. 2017. 
# ------------------------------------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from Net_Structure.BatchNorm import BatchNormLayerForCNN
from Net_Structure.ResidualBlock import ResidualBlock
from Net_Structure.BottleNeck import BottleNeck

import sys
sys.path.append('/tigress/yl16/Third_Party_Code/AdaBound-Tensorflow')
from AdaBound import AdaBoundOptimizer

# CONSTANTS
CONV_KER_SIZE = 3
INP_MAP = 3

def weight_variable(shape,_name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=_name)
    return tf.Variable(initial)

def bias_variable(shape,_name):
    initial = tf.constant(0.0, shape=shape,name=_name)
    return tf.Variable(initial)
    
class ResNet:
    
    #------------------------- A Very Long Model Initialization Part -------------------------
    def __init__(self, block_type, num_res_blocks = 9, param_list = None, BN_Var_Epsilon = 1e-5, BN_Momentum = 0.99, Dataset ='CIFAR100'):

        '''
        Usage:
            Create a ResNet object
        Args:	
	    block_type:     The type of the block to construct the ResNet
            num_res_blocks: Scalar number of residual blocks in each feature map stage (characterized by size)
            BN_Var_Epsilon: The var epsilon for BN
            BN_Momentum:    The momentum for BN
            param_list:     A list of numpy array to store the parameters
            Dataset:        A string indicating which dataset to train for the network
        Returns:
            A ResNet object
        '''

        assert block_type in ['Normal', 'BottleNeck']
        self.block_type = block_type

        if block_type == 'Normal':
            self.blk = ResidualBlock
            N_CONV_MAP_1, N_CONV_MAP_2, N_CONV_MAP_3 = (16,32,64)
            
        if block_type == 'BottleNeck':
            self.blk = BottleNeck
            N_CONV_MAP_1, N_CONV_MAP_2, N_CONV_MAP_3 = (64,128,256)

        self.num_res_blocks = num_res_blocks
        
        if param_list == None:
            # The initial convolution layer
            self.W_conv0 = weight_variable([CONV_KER_SIZE, CONV_KER_SIZE, INP_MAP, N_CONV_MAP_1], 'W_conv0')
            self.batch_norm0 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, num_channel = N_CONV_MAP_1)

            ''' The Parameters for Residual Block'''
            # Feature Map size 32 * 32
            self.block_list_32 = []
            for i in range(num_res_blocks):
                if i == 0:
                    res_block1 = self.blk(inp_chan = N_CONV_MAP_1, out_chan = N_CONV_MAP_1, is_first_block =True, 
                                               BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                else:
                    res_block1 = self.blk(inp_chan = N_CONV_MAP_1, out_chan = N_CONV_MAP_1, BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                self.block_list_32.append(res_block1)

            # Feature Map size 16 * 16
            self.block_list_16 = []            
            for i in range(num_res_blocks):
                if i == 0:
                    res_block2 = self.blk(inp_chan = N_CONV_MAP_1, out_chan = N_CONV_MAP_2, downsample = True,
					       BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                else:
                    res_block2 = self.blk(inp_chan = N_CONV_MAP_2, out_chan =  N_CONV_MAP_2, BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                self.block_list_16.append(res_block2)

            # Feature Map size 8 * 8
            self.block_list_8 = []            
            for i in range(num_res_blocks):
                if i == 0:
                    res_block3 = self.blk(inp_chan = N_CONV_MAP_2, out_chan = N_CONV_MAP_3, downsample = True,
					       BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                else:
                    res_block3 = self.blk(inp_chan = N_CONV_MAP_3, out_chan = N_CONV_MAP_3, BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                self.block_list_8.append(res_block3)

            # Final Fully Connected Layer
            assert Dataset in ['CIFAR10', 'CIFAR100']
            if Dataset == 'CIFAR100':
                output_dim = 100
            elif Dataset == 'CIFAR10':
                output_dim = 10

            self.W_fc0 = weight_variable([N_CONV_MAP_3, output_dim], 'W_fc0')
            self.b_fc0 = bias_variable([output_dim], 'b_fc0')
            self.batch_normfc = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, num_channel = N_CONV_MAP_3)

        else:
            ''' 
            Loaded from external model
            '''

            # Extract the paramlist for each portion of the network first
            first_layer_param, blk_param, final_layer_param = param_list[0], param_list[1:-1], param_list[-1]

            # First convolutional layer
            self.W_conv0 = tf.Variable(first_layer_param[0], name = 'W_conv0')
            self.batch_norm0 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, BN_param_list = first_layer_param[1])

            ''' 
            The Parameters for Residual Block
            '''

            # Feature Map size 32 * 32
            iter_ind = 0
            self.block_list_32 = []
            for i in range(num_res_blocks):
                if i == 0:
                    res_block1 = self.blk(inp_chan = N_CONV_MAP_1, out_chan = N_CONV_MAP_1, is_first_block =True, 
                                               BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum, param_list = blk_param[iter_ind])
                else:
                    res_block1 = self.blk(inp_chan = N_CONV_MAP_1, out_chan = N_CONV_MAP_1, param_list = blk_param[iter_ind], 
					       BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                self.block_list_32.append(res_block1)
                iter_ind = iter_ind + 1

            # Feature Map size 16 * 16
            self.block_list_16 = []            
            for i in range(num_res_blocks):
                if i == 0:
                    res_block2 = self.blk(inp_chan = N_CONV_MAP_1, out_chan = N_CONV_MAP_2, downsample = True,
					       BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum, param_list = blk_param[iter_ind])
                else:
                    res_block2 = self.blk(inp_chan = N_CONV_MAP_2, out_chan = N_CONV_MAP_2, param_list = blk_param[iter_ind],
 					       BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                self.block_list_16.append(res_block2)
                iter_ind = iter_ind + 1

            # Feature Map size 8 * 8
            self.block_list_8 = []            
            for i in range(num_res_blocks):
                if i == 0:
                    res_block3 = self.blk(inp_chan = N_CONV_MAP_2, out_chan = N_CONV_MAP_3, downsample = True,
					       BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum, param_list = blk_param[iter_ind])
                else:
                    res_block3 = self.blk(inp_chan = N_CONV_MAP_3, out_chan = N_CONV_MAP_3, param_list = blk_param[iter_ind],
					       BN_Var_Epsilon = BN_Var_Epsilon, BN_Momentum = BN_Momentum)
                self.block_list_8.append(res_block3)
                iter_ind = iter_ind + 1
            

            # Last fully connected layer 
            self.W_fc0 = tf.Variable(final_layer_param[0], name = 'W_fc0')
            self.b_fc0 = tf.Variable(final_layer_param[1], name = 'b_fc0')
            self.batch_normfc = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, BN_param_list = final_layer_param[2])

        # The Overall parameters, Defined for Variable Initialization
        self.non_block_param =  [self.W_conv0, self.W_fc0, self.b_fc0]
        self.block_list = [self.block_list_32, self.block_list_16, self.block_list_8]                                    
        self.non_block_bn = [self.batch_norm0, self.batch_normfc]

        # This bn list is defined just for training op update!
        self.bn_list = [self.batch_norm0]
        for res_list in self.block_list:
            for res in res_list:
                self.bn_list += res.bn_list
        self.bn_list.append(self.batch_normfc)


    def inference(self, x_image, is_training=False, res_mask_list = None):
        
        '''
        Usage:
            Map a 4D image tensor of shape [?, 224, 224, 3] to a logit [? , 1000]
        Args:
            x_image:       The image to be inferenced
            is_training:   Default to be False which means in the testing mode
            res_mask_list: The mask list for each layer for pruning
        Returns:
            layer_prune:   The dictionary contains the target layers for pruning
            logits:        A 2D tensor with shape [?, 1000]
        '''        

        # Initialize them for mask free operation
        if res_mask_list == None:
            res_mask_list = [[None] * self.num_res_blocks] * 3 # totally three fea map size

        out_layers = []
        layer_prune = {}

        # The first conv layer out of the block
        h0 = tf.nn.conv2d(x_image, self.W_conv0, strides=[1, 1, 1, 1], padding='SAME')
        h_BN0 = self.batch_norm0.feed_forward(h0,is_training)
        h_relu0 = tf.nn.relu(h_BN0)

        out_layers.append(h_relu0)

        # Res32 block
        for i in range(self.num_res_blocks):
            res32 = self.block_list_32[i]
            res_layer, res_out = res32.inference(out_layers[-1], is_training, res_mask_list[0][i])
            layer_prune['Res32_' + str(i)] = res_layer
            out_layers.append(res_out)

        # Res16 block
        for i in range(self.num_res_blocks):
            res16 = self.block_list_16[i]
            res_layer, res_out = res16.inference(out_layers[-1], is_training, res_mask_list[1][i])
            layer_prune['Res16_' + str(i)] = res_layer
            out_layers.append(res_out)

        # Res8 block
        for i in range(self.num_res_blocks):
            res8 = self.block_list_8[i]
            res_layer, res_out = res8.inference(out_layers[-1], is_training, res_mask_list[2][i])
            layer_prune['Res8_' + str(i)] = res_layer
            out_layers.append(res_out)

        # Fully Connected Layers
        with tf.name_scope('fc'):
            h_BNfc = self.batch_normfc.feed_forward(out_layers[-1],is_training)
            h_relufc = tf.nn.relu(h_BNfc)
            h_pool = tf.reduce_mean(h_relufc, [1, 2]) # global average pooling
            logits = tf.matmul(h_pool,self.W_fc0) + self.b_fc0

        return [layer_prune, logits]

    def loss(self, logits, labels, ridge):

        '''
        Usage:
            Defines the loss function for training the neural network
        Args:
            logits: The logit after network inference
            labels: The ground truth labels
            ridge:  The l2 ridge parameter for weight decay for the whole network
        Returns:
            CE_loss:    The cross entropy loss
            ridge_loss: The weight decay loss
        '''

        labels = tf.to_int64(labels)
        CE_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        fc_l2 = tf.nn.l2_loss(self.W_fc0) + tf.nn.l2_loss(self.b_fc0)
        conv_l2 = tf.nn.l2_loss(self.W_conv0)
        for res_list in self.block_list:
            for res in res_list:
                conv_l2 = conv_l2 + tf.nn.l2_loss(res.W_conv1) + tf.nn.l2_loss(res.W_conv2)
                if self.block_type == 'BottleNeck': # Additional weight param
            	    conv_l2 = conv_l2 + tf.nn.l2_loss(res.W_conv3)
        ridge_loss = ridge * (fc_l2 + conv_l2)

        return [CE_loss,ridge_loss]

    def training(self, opt_name, loss, learning_rate, momentum = None, final_lr = None):

        '''
        Usage:
            Define the training operation in a single-gpu settings
        
        Args:
            opt_name:       The optimizer to use
            loss:           The loss op for optimization
            learning_rate:  Learning rate
            momentum:       Momentum for the optimizer if the optimizer requires momentum
        Returns:
            The operation for training the neural network
        '''

        assert opt_name in ['Nestrov_Momentum','Adam','Momentum','AdaBound']

        if opt_name == 'Nestrov_Momentum':
            assert np.isscalar(momentum)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        elif opt_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif opt_name == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        elif opt_name == 'AdaBound':
            optimizer = AdaBoundOptimizer(learning_rate = learning_rate, final_lr = final_lr)

        # Use the optimizer to apply the gradients that minimize the loss
        apply_grad_op = optimizer.minimize(loss = loss)

	# Get the update operation for bn moving mean and variance
        bn_update_op = []
        for bn in self.bn_list:
            bn_update_op += bn.update_moving_mean_var()

        train_op = [apply_grad_op, bn_update_op]
        return train_op

    def model_save(self, m_sess):
        '''
        Usage:
            Store the value in tf Variable to a numpy array
        Args:
            m_sess: the tf Session to run operation
        Returns:
            the parameter of the whole network
        '''

        conv0_param = m_sess.run([self.W_conv0, self.batch_norm0.param_list]) 
        fc_param = m_sess.run([self.W_fc0, self.b_fc0, self.batch_normfc.param_list]) 
        block_param = []
        for res_list in self.block_list:
            for block in res_list:
                block_param.append(block.block_save(m_sess))

        total_param = [conv0_param] + block_param + [fc_param]
        return total_param


    def model_load(self, param_list, m_sess):

        '''
        Usage:
            Load the value of a numpy array list into the Variable of the network
        Args:
            param_list: The list of numpy array contains values to be loaded
            m_sess:     The tf session to run operations
        '''

        # load the param for 1st conv0 layer
        conv0_assign_op = [tf.assign(self.W_conv0, param_list[0][0]), self.batch_norm0.bn_load(param_list[0][1])]

        # load the param for last fc layer
        fc_assign_op = [tf.assign(self.W_fc0, param_list[-1][0]), tf.assign(self.b_fc0, param_list[-1][1]), self.batch_normfc.bn_load(param_list[-1][2])]

        # load the param for the residual block
        block_ind = 1
        block_assign_op = []
        for res_list in self.block_list:
            for block in res_list:
                block_assign_op.append(block.block_load(param_list[block_ind]))
                block_ind = block_ind + 1

        # run these assignments opeartions
        m_sess.run([conv0_assign_op, fc_assign_op, block_assign_op])
       

    def model_initialization(self, m_sess): 
        '''
        Usage:
            Locally initialize all the parameters in the network
        
        Args:
            m_sess: The tf session to run the operations
        '''

        # Initialize Weight
        non_block_param_init_list = []
        for var in self.non_block_param:
            non_block_param_init_list.append(var.initializer)
        
        # Initialize BN
        non_block_bn_init_list = []
        for bn_var in self.non_block_bn:
            non_block_bn_init_list.append(bn_var.var_initialization())

        # Initialize the Block
        blk_init_list = []
        for blk_list in self.block_list:
            for blk in blk_list:
                blk_init_list.append(blk.block_initialization())
        
        m_sess.run([non_block_param_init_list, non_block_bn_init_list, blk_init_list])

    
    def compressed_model(self, mask_list, m_sess):
        '''
        Usage:
            Compress the parameters of the whole network by a given mask list
        Args:
            mask_list: A list of mask to indicating the nodes to prune in each layer
            m_sess:    The session to run the operation
        Returns:
            A list of numpy array containing values of the compressed network
        '''

        model_param = self.model_save(m_sess)
        conv0_param, block_param, fc_param = model_param[0], model_param[1:-1], model_param[-1]
                
        # compress the param for the residual block
        comp_block_param = []
        for i in range(3):
            for j in range(self.num_res_blocks):
                mask = mask_list[i][j]
                comp_block = self.block_list[i][j].block_compression(mask,m_sess) 
                comp_block_param.append(comp_block)
        
        compressed_ResNet = [conv0_param] + comp_block_param + [fc_param]
        
        return compressed_ResNet


    def net_get_mask_pl_list(self):
        '''
        Usage:
            Get the placeholder for layer masking
        Returns:
            the placeholder for layer masking
        '''

        net_mask_pl = []
        for blk_list in self.block_list:
            blk_mask_pl_list = []
            for blk in blk_list:
                blk_mask_pl_list.append(blk.block_get_mask_pl_list())
            net_mask_pl.append(blk_mask_pl_list)
        return net_mask_pl


