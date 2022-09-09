# ------------------------------------------------------------------------------------------------------------
# This Code Implements Residual Block for ResNet in TensorFlow
# Reference code: https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
# Reference paper: He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
# ------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
from Net_Structure.BatchNorm import BatchNormLayerForCNN

# CONSTANTS
CONV_KER_SIZE_3 = 3
CONV_KER_SIZE_1 = 1

def weight_variable(shape,_name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=_name)
    return tf.Variable(initial)

class BottleNeck:
    def __init__(self, inp_chan, out_chan, downsample = False, BN_Var_Epsilon = 1e-5, BN_Momentum = 0.99, is_first_block = False, param_list = None):
        '''
        Usage:
            Defines a residual block in ResNet
        Args:
            inp_chan: number of input channel
            out_chan: number of output channel
            downsample: whether the block performs downsampling by stride 2 convolution or not         
            first_block: if this is the first residual block of the whole network
            param_list:  A list of numpy array values to initialize the parameters
        Returns:
            A residual block object 
        '''
        
        self.downsample = downsample
        self.is_first_block = is_first_block
        self.inp_chan = inp_chan
        self.out_chan = out_chan

        if param_list == None:
            bot_chan = out_chan // 4
            self.W_conv1 = weight_variable([CONV_KER_SIZE_1, CONV_KER_SIZE_1, inp_chan, bot_chan], 'W_conv1')
            self.W_conv2 = weight_variable([CONV_KER_SIZE_3, CONV_KER_SIZE_3, bot_chan, bot_chan], 'W_conv2')
            self.W_conv3 = weight_variable([CONV_KER_SIZE_1, CONV_KER_SIZE_1, bot_chan, out_chan], 'W_conv3')

            if is_first_block:
                self.batch_norm2 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, num_channel = bot_chan)          
                self.batch_norm3 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, num_channel = bot_chan)
                self.bn_list = [self.batch_norm2, self.batch_norm3]
            else:
                self.batch_norm1 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, num_channel = inp_chan)
                self.batch_norm2 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, num_channel = bot_chan)
                self.batch_norm3 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, num_channel = bot_chan)
                self.bn_list = [self.batch_norm1, self.batch_norm2, self.batch_norm3]
            self.inp_sel = [True] * inp_chan

        else:
            weight_param, bn_param, inp_sel = param_list[:3], param_list[3:-1], param_list[-1]
            self.W_conv1 = tf.Variable(weight_param[0], name = 'W_conv1')
            self.W_conv2 = tf.Variable(weight_param[1], name = 'W_conv2')
            self.W_conv3 = tf.Variable(weight_param[2], name = 'W_conv3')
            if is_first_block:
                self.batch_norm2 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, BN_param_list = bn_param[0])
                self.batch_norm3 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, BN_param_list = bn_param[1])
                self.bn_list = [self.batch_norm2, self.batch_norm3]
            else:
                self.batch_norm1 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, BN_param_list = bn_param[0])
                self.batch_norm2 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, BN_param_list = bn_param[1])
                self.batch_norm3 = BatchNormLayerForCNN(Var_Epsilon = BN_Var_Epsilon, Momentum = BN_Momentum, BN_param_list = bn_param[2])
                self.bn_list = [self.batch_norm1, self.batch_norm2, self.batch_norm3]
            self.inp_sel = inp_sel
            
        self.weight_list = [self.W_conv1, self.W_conv2, self.W_conv3]


        
    def inference(self, input_tensor, is_training = False, conv_mask_list = None):        
        '''
        Usage:
            Defines the inference path of the residual block
        
        Args:
            inp_tensor:     Input tensor to the block
            is_training:    The inference mode of the block, False means in evaluation mode
            conv_mask_list: The list of mask which are used to zero the channel activation to achieve pruning 
        Returns:
            layer_dict:    The dictionary containing layer to be pruned
            output_tensor: The output tensor of the block
        '''

        
        if conv_mask_list == None:
            conv1_mask,conv2_mask,conv3_mask = [[True] * W.shape[-2] for W in self.weight_list]

        else:
            conv1_mask,conv2_mask,conv3_mask = conv_mask_list

        # When it's time to "shrink" the image size, we use stride = 2 for downsampling
        if self.downsample:
            stride = 2
        else: 
            stride = 1

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
           
            sel_ind = [i for i in range(input_tensor.shape[-1]) if self.inp_sel[i]]
            inp_sel_layer = tf.gather(params=input_tensor, indices=sel_ind, axis=-1)            

            if self.is_first_block:
                act1_post_mask = tf.multiply(inp_sel_layer, conv1_mask)
                conv1 = tf.nn.conv2d(act1_post_mask, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            else:
                bn1 = self.batch_norm1.feed_forward(inp_sel_layer,is_training)
                act1 = tf.nn.relu(bn1)
                act1_post_mask = tf.multiply(act1,conv1_mask)
                conv1 = tf.nn.conv2d(act1_post_mask, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
                
        with tf.variable_scope('conv2_in_block'):
            bn2 = self.batch_norm2.feed_forward(conv1,is_training)
            act2 = tf.nn.relu(bn2)
            act2_post_mask = tf.multiply(act2,conv2_mask)
            conv2 = tf.nn.conv2d(act2_post_mask, self.W_conv2, strides=[1, stride, stride, 1], padding='SAME')

        with tf.variable_scope('conv3_in_block'):
            bn3 = self.batch_norm3.feed_forward(conv2,is_training)
            act3 = tf.nn.relu(bn3)
            act3_post_mask = tf.multiply(act3,conv3_mask)
            conv3 = tf.nn.conv2d(act3_post_mask, self.W_conv3, strides=[1, 1, 1, 1], padding='SAME')

        layer_dict = {'Relu1 Output': act1_post_mask, 'Relu2 Output': act2_post_mask, 'Relu3 Output': act3_post_mask}

        # When the channels dimension of input layer and output conv2 does not match, we pool the input and pad zero channels to match the dimensionality
        with tf.variable_scope('shortcut_connection'):
            if self.downsample:
                channel_diff = self.out_chan - self.inp_chan
                pooled_input = tf.nn.avg_pool(input_tensor, ksize=[1, 2, 2, 1], 
                                            strides=[1, 2, 2, 1], padding='VALID')
                padded_input = tf.pad(pooled_input, [ [0, 0], [0, 0], [0, 0], 
                                [int(np.floor(channel_diff/2.0)), int(np.ceil(channel_diff/2.0))] ])
            else:
                padded_input = input_tensor
            
            output_tensor = conv3 + padded_input

        return [layer_dict, output_tensor]



    def block_save(self,m_sess):
        '''
        Usage:
            Store the TensorFlow Variable to numpy array
        
        Args:
            m_sess: The session use to run the operation
        Returns:
            A list of numpy array containing the parameter values
        '''

        #weight_param = m_sess.run(self.weight_list)
        #bn_param = [m_sess.run(bn.param_list) for bn in self.bn_list]

        weight_bn_param = m_sess.run(self.weight_list + [bn.param_list for bn in self.bn_list])
        block_param = weight_bn_param + [self.inp_sel]
        return block_param



    def block_load(self,param_list):
        '''
        Usage:
            Load the parameter from numpy into the block params
        Args:
            param_list: A value list of numpy array for the block parameters
        Returns:
            The list of the assigning operations
        '''

        weight_param, bn_param, inp_sel = param_list[:3], param_list[3:-1], param_list[-1]

        weight_assign_ops = []
        for w,w_tf in zip(weight_param,self.weight_list):
            weight_assign_ops.append(tf.assign(w_tf,w))

        bn_assign_ops = []
        for bn,bn_tf in zip(bn_param,self.bn_list):
            bn_assign_ops.append(bn_tf.bn_load(bn))

        self.inp_sel = inp_sel

        return [weight_assign_ops,bn_assign_ops]


    def block_initialization(self):
        '''
        Usage:
            Locally initialize the block
        Returns:
            Initialization List
        '''
        
        weight_init_list = []
        for var in self.weight_list:
            weight_init_list.append(var.initializer)
        
        bn_var_init_list = []
        for bn_var in self.bn_list:
            bn_var_init_list.append(bn_var.var_initialization())
            
        return [weight_init_list,bn_var_init_list]

### ============================================== Method Defined for Pruning ==============================================
    # Compressing a block
    def block_compression(self, mask_list, m_sess):
        conv1_mask, conv2_mask, conv3_mask = mask_list
        blk_param = self.block_save(m_sess)
        weight_param, bn_param, inp_sel = blk_param[:len(self.weight_list)], blk_param[len(self.weight_list):-1], blk_param[-1]

        # Compressed the inp_sel
        comp_inp_sel = np.array(list(inp_sel))
        cur_ind = np.where(np.array(inp_sel) == True)[0]
        mask_ind = np.where(np.array(conv1_mask) == False)[0]
        ind_to_mask = cur_ind[mask_ind]
        comp_inp_sel[ind_to_mask] = False
        comp_inp_sel = list(comp_inp_sel)

        # Compressed the weight param
        W1,W2,W3 = weight_param
        comp_W1 = W1[:,:,conv1_mask,:][:,:,:,conv2_mask]
        comp_W2 = W2[:,:,conv2_mask,:][:,:,:,conv3_mask]
        comp_W3 = W3[:,:,conv3_mask,:]
        comp_weight_param = [comp_W1,comp_W2,comp_W3]


        # Compressed the BN param
        if self.is_first_block:
            bn2,bn3 = bn_param
            comp_bn2 = [bn_p[conv2_mask] for bn_p in bn2]
            comp_bn3 = [bn_p[conv3_mask] for bn_p in bn3]
            comp_bn_param = [comp_bn2,comp_bn3]
        else:
            bn1,bn2,bn3 = bn_param
            comp_bn1 = [bn_p[conv1_mask] for bn_p in bn1]
            comp_bn2 = [bn_p[conv2_mask] for bn_p in bn2]
            comp_bn3 = [bn_p[conv3_mask] for bn_p in bn3]
            comp_bn_param = [comp_bn1,comp_bn2,comp_bn3]

        comp_param = comp_weight_param + comp_bn_param + [comp_inp_sel]
        return comp_param


    def block_get_mask_pl_list(self):
        block_mask_pl = [tf.placeholder(tf.float32, shape=[W.shape[-2]]) for W in (self.weight_list)]
        return block_mask_pl
