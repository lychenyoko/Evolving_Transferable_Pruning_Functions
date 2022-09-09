import numpy as np
import tensorflow as tf

'''Only these params Differ'''
conv_ker_size = 3

inp_map = 3
conv_map_1 = 64
conv_map_2 = 128
conv_map_3 = 256
conv_map_4 = 512

fla_dim = 512
fc_1_dim = 512
output_dim = 10

BN_var_epsilon = 1e-3

#####--------------------------------- The above part will not copy to Tiger ----------------------------------

def weight_variable(shape,_name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=_name)
    return tf.Variable(initial)

def bias_variable(shape,_name):
    initial = tf.constant(0.0, shape=shape,name=_name)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')


class BatchNormLayerForCNN:
    def __init__(self, num_channel, BN_param_list = None):

        if BN_param_list == None:
            self.inference_mean = tf.Variable(tf.zeros(num_channel),trainable = False)
            self.inference_var = tf.Variable(tf.ones(num_channel), trainable = False)
            self.offset = tf.Variable(tf.zeros(num_channel))
            self.scale = tf.Variable(tf.ones(num_channel))            
        else:
            self.scale = tf.Variable(BN_param_list[0])    
            self.offset = tf.Variable(BN_param_list[1])            
            self.inference_mean = tf.Variable(BN_param_list[2],trainable = False)
            self.inference_var = tf.Variable(BN_param_list[3],trainable = False)

        self.param_list = [self.scale,self.offset,self.inference_mean,self.inference_var]

    def var_initialization(self,sess):
        for var in self.param_list:
            sess.run(var.initializer)

    def feed_forward(self,fea_map_input,is_training,momentum):
        if is_training == True:
            flated_inp = tf.reshape(fea_map_input,[-1,fea_map_input.shape[-1]])
            batch_mean,batch_var = tf.nn.moments(flated_inp,[0])

            # Output feature map tensor and update the inference mean            
            fea_map_output = tf.nn.batch_normalization(fea_map_input, batch_mean, batch_var, self.offset, self.scale, BN_var_epsilon)

            # Self update operation
            self.moving_mean = tf.assign(self.inference_mean, self.inference_mean * momentum + (1 - momentum) * batch_mean)
            self.moving_var = tf.assign(self.inference_var, self.inference_var * momentum + (1 - momentum) * batch_var)

            return fea_map_output
        else:
            fea_map_output = tf.nn.batch_normalization(
                fea_map_input, self.inference_mean, self.inference_var, self.offset, self.scale, BN_var_epsilon
            )
            return fea_map_output

    def bn_recovery(self,param_list,m_sess):
        bn_assign_op = []
        for self_param, param in zip(self.param_list,param_list):
            bn_assign_op.append(tf.assign(self_param,param))
        m_sess.run(bn_assign_op)    
        

class VGG16:
   
    #------------------------- A Very Long Model Initialization Part -------------------------
    def __init__(self, param_list = None):
        
        if param_list == None:
            # Feature Map size 32 * 32
            self.W_conv1 = weight_variable([conv_ker_size, conv_ker_size, inp_map, conv_map_1], 'W_conv1')
            self.b_conv1 = bias_variable([conv_map_1], 'b_conv1')
            self.batch_norm1 = BatchNormLayerForCNN(conv_map_1)
            self.W_conv2 = weight_variable([conv_ker_size, conv_ker_size, conv_map_1, conv_map_1], 'W_conv2')
            self.b_conv2 = bias_variable([conv_map_1], 'b_conv2')
            self.batch_norm2 = BatchNormLayerForCNN(conv_map_1)            
            
            # Feature Map size 16 * 16
            self.W_conv3 = weight_variable([conv_ker_size, conv_ker_size, conv_map_1, conv_map_2], 'W_conv3')
            self.b_conv3 = bias_variable([conv_map_2], 'b_conv3')
            self.batch_norm3 = BatchNormLayerForCNN(conv_map_2)            
            self.W_conv4 = weight_variable([conv_ker_size, conv_ker_size, conv_map_2, conv_map_2], 'W_conv4')
            self.b_conv4 = bias_variable([conv_map_2], 'b_conv4')
            self.batch_norm4 = BatchNormLayerForCNN(conv_map_2)                        
            
            # Feature Map size 8 * 8
            self.W_conv5 = weight_variable([conv_ker_size, conv_ker_size, conv_map_2, conv_map_3], 'W_conv5')
            self.b_conv5 = bias_variable([conv_map_3], 'b_conv5')
            self.batch_norm5 = BatchNormLayerForCNN(conv_map_3)                        
            self.W_conv6 = weight_variable([conv_ker_size, conv_ker_size, conv_map_3, conv_map_3], 'W_conv6')
            self.b_conv6 = bias_variable([conv_map_3], 'b_conv6')
            self.batch_norm6 = BatchNormLayerForCNN(conv_map_3)                        
            self.W_conv7 = weight_variable([conv_ker_size, conv_ker_size, conv_map_3, conv_map_3], 'W_conv7')
            self.b_conv7 = bias_variable([conv_map_3], 'b_conv7')
            self.batch_norm7 = BatchNormLayerForCNN(conv_map_3)                                    
            
            # Feature Map size 4 * 4
            self.W_conv8 = weight_variable([conv_ker_size, conv_ker_size, conv_map_3, conv_map_4], 'W_conv8')
            self.b_conv8 = bias_variable([conv_map_4], 'b_conv8')
            self.batch_norm8 = BatchNormLayerForCNN(conv_map_4)                                                              
            self.W_conv9 = weight_variable([conv_ker_size, conv_ker_size, conv_map_4, conv_map_4], 'W_conv9')
            self.b_conv9 = bias_variable([conv_map_4], 'b_conv9')
            self.batch_norm9 = BatchNormLayerForCNN(conv_map_4)                                                                          
            self.W_conv10 = weight_variable([conv_ker_size, conv_ker_size, conv_map_4, conv_map_4], 'W_conv10')
            self.b_conv10 = bias_variable([conv_map_4], 'b_conv10') 
            self.batch_norm10 = BatchNormLayerForCNN(conv_map_4)            
            
            # Feature Map size 2 * 2
            self.W_conv11 = weight_variable([conv_ker_size, conv_ker_size, conv_map_4, conv_map_4], 'W_conv11')
            self.b_conv11 = bias_variable([conv_map_4], 'b_conv11')
            self.batch_norm11 = BatchNormLayerForCNN(conv_map_4)                                      
            self.W_conv12 = weight_variable([conv_ker_size, conv_ker_size, conv_map_4, conv_map_4], 'W_conv12')
            self.b_conv12 = bias_variable([conv_map_4], 'b_conv12')
            self.batch_norm12 = BatchNormLayerForCNN(conv_map_4)            
            self.W_conv13 = weight_variable([conv_ker_size, conv_ker_size, conv_map_4, conv_map_4], 'W_conv13')
            self.b_conv13 = bias_variable([conv_map_4], 'b_conv13') 
            self.batch_norm13 = BatchNormLayerForCNN(conv_map_4)            
            
            # Fully Connected Layer
            self.W_fc1 = weight_variable([fla_dim, fc_1_dim], 'W_fc1')
            self.b_fc1 = bias_variable([fc_1_dim], 'b_fc1')
            self.batch_norm14 = BatchNormLayerForCNN(fc_1_dim)
            self.W_fc2 = weight_variable([fc_1_dim, output_dim], 'W_fc2')
            self.b_fc2 = bias_variable([output_dim], 'b_fc2')
        
        
        else: # Loaded from external model
            # Convolutional Layer
            self.W_conv1 = tf.Variable(param_list[0], name = 'W_conv1')
            self.b_conv1 = tf.Variable(param_list[1], name = 'b_conv1')
            self.W_conv2 = tf.Variable(param_list[2], name = 'W_conv2')
            self.b_conv2 = tf.Variable(param_list[3], name = 'b_conv2')
            
            # Convolutional Layer
            self.W_conv3 = tf.Variable(param_list[4], name = 'W_conv3')
            self.b_conv3 = tf.Variable(param_list[5], name = 'b_conv3')
            self.W_conv4 = tf.Variable(param_list[6], name = 'W_conv4')
            self.b_conv4 = tf.Variable(param_list[7], name = 'b_conv4')
            
            # Convolutional Layer
            self.W_conv5 = tf.Variable(param_list[8], name = 'W_conv5')
            self.b_conv5 = tf.Variable(param_list[9], name = 'b_conv5')
            self.W_conv6 = tf.Variable(param_list[10], name = 'W_conv6')
            self.b_conv6 = tf.Variable(param_list[11], name = 'b_conv6')
            
            # Convolutional Layer
            self.W_conv7 = tf.Variable(param_list[12], name = 'W_conv7')
            self.b_conv7 = tf.Variable(param_list[13], name = 'b_conv7')
            self.W_conv8 = tf.Variable(param_list[14], name = 'W_conv8')
            self.b_conv8 = tf.Variable(param_list[15], name = 'b_conv8')
            
            # Convolutional Layer
            self.W_conv9 = tf.Variable(param_list[16], name = 'W_conv9')
            self.b_conv9 = tf.Variable(param_list[17], name = 'b_conv9')
            self.W_conv10 = tf.Variable(param_list[18], name = 'W_conv10')
            self.b_conv10 = tf.Variable(param_list[19], name = 'b_conv10')            

            # Convolutional Layer
            self.W_conv11 = tf.Variable(param_list[20], name = 'W_conv11')
            self.b_conv11 = tf.Variable(param_list[21], name = 'b_conv11')
            self.W_conv12 = tf.Variable(param_list[22], name = 'W_conv12')
            self.b_conv12 = tf.Variable(param_list[23], name = 'b_conv12')
            
            # Convolutional Layer
            self.W_conv13 = tf.Variable(param_list[24], name = 'W_conv13')
            self.b_conv13 = tf.Variable(param_list[25], name = 'b_conv13')

            # Fully Connected Layer
            self.W_fc1 = tf.Variable(param_list[26], name = 'W_fc1')
            self.b_fc1 = tf.Variable(param_list[27], name = 'b_fc1')
            self.W_fc2 = tf.Variable(param_list[28], name = 'W_fc2')
            self.b_fc2 = tf.Variable(param_list[29], name = 'b_fc2')
            
            # Batch Norm Parameters
            self.batch_norm1 = BatchNormLayerForCNN(self.b_conv1.shape[0],param_list[30])
            self.batch_norm2 = BatchNormLayerForCNN(self.b_conv2.shape[0],param_list[31])
            self.batch_norm3 = BatchNormLayerForCNN(self.b_conv3.shape[0],param_list[32])
            self.batch_norm4 = BatchNormLayerForCNN(self.b_conv4.shape[0],param_list[33])
            self.batch_norm5 = BatchNormLayerForCNN(self.b_conv5.shape[0],param_list[34])
            self.batch_norm6 = BatchNormLayerForCNN(self.b_conv6.shape[0],param_list[35])
            self.batch_norm7 = BatchNormLayerForCNN(self.b_conv7.shape[0],param_list[36])
            self.batch_norm8 = BatchNormLayerForCNN(self.b_conv8.shape[0],param_list[37])
            self.batch_norm9 = BatchNormLayerForCNN(self.b_conv9.shape[0],param_list[38])
            self.batch_norm10 = BatchNormLayerForCNN(self.b_conv10.shape[0],param_list[39])
            self.batch_norm11 = BatchNormLayerForCNN(self.b_conv11.shape[0],param_list[40])
            self.batch_norm12 = BatchNormLayerForCNN(self.b_conv12.shape[0],param_list[41])
            self.batch_norm13 = BatchNormLayerForCNN(self.b_conv13.shape[0],param_list[42])
            self.batch_norm14 = BatchNormLayerForCNN(self.b_fc1.shape[0],param_list[43])


        self.network_param_list =  [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                    self.W_conv3, self.b_conv3, self.W_conv4, self.b_conv4,
                    self.W_conv5, self.b_conv5, self.W_conv6, self.b_conv6,
                    self.W_conv7, self.b_conv7, self.W_conv8, self.b_conv8,
                    self.W_conv9, self.b_conv9, self.W_conv10, self.b_conv10,
                    self.W_conv11, self.b_conv11, self.W_conv12, self.b_conv12,
                    self.W_conv13, self.b_conv13,
                    self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

        self.batch_norm_list = [self.batch_norm1,self.batch_norm2,self.batch_norm3,self.batch_norm4,
                                self.batch_norm5,self.batch_norm6,self.batch_norm7,self.batch_norm8,
                                self.batch_norm9,self.batch_norm10,self.batch_norm11,self.batch_norm12,
                                self.batch_norm13,self.batch_norm14]                 

    # Initialze Varaibale for the network
    def var_initialization(self, m_sess): 
        for var in self.network_param_list:
            m_sess.run(var.initializer)
        for bn_var in self.batch_norm_list:
            bn_var.var_initialization(m_sess)
        
    
    def get_mask_pl_list(self):
        mask_pl_list = [tf.placeholder(tf.float32, shape = [bias.shape[0]]) for bias in self.network_param_list[1:-2:2]]
        return mask_pl_list

    # Gonna be a large part!
    def inference(self, x_image, is_training, keep_prob, mask_list = None):
        
        # Initialize them for mask free operation
        if mask_list == None:
            conv1_mask,conv2_mask,conv3_mask,conv4_mask,conv5_mask, \
            conv6_mask, conv7_mask,conv8_mask,conv9_mask,conv10_mask, \
            conv11_mask, conv12_mask, conv13_mask,afc1_mask \
            = [[True] * bias.shape[0] for bias in self.network_param_list[1:-2:2]]

        else:
            conv1_mask,conv2_mask,conv3_mask,conv4_mask,conv5_mask, \
            conv6_mask, conv7_mask,conv8_mask,conv9_mask,conv10_mask, \
            conv11_mask, conv12_mask, conv13_mask,afc1_mask = mask_list


        # ------------------------ building the graph here ------------------------

        # Feature Map Size 32 * 32
        with tf.name_scope('conv1'):
            h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
            h_BN1 = self.batch_norm1.feed_forward(h_conv1,is_training,0.99)
            h_dropout1 = tf.nn.dropout(h_BN1,keep_prob)
            h_conv1_post_mask = tf.multiply(h_dropout1,conv1_mask)

        with tf.name_scope('conv2'):
            h_conv2 = tf.nn.relu(conv2d(h_conv1_post_mask, self.W_conv2) + self.b_conv2)
            h_BN2 = self.batch_norm2.feed_forward(h_conv2,is_training,0.99)            
            h_pool2 = max_pool_2x2(h_BN2)
            h_pool2_post_mask = tf.multiply(h_pool2,conv2_mask)

        # Feature Map Size 16 * 16
        with tf.name_scope('conv3'):
            h_conv3 = tf.nn.relu(conv2d(h_pool2_post_mask, self.W_conv3) + self.b_conv3)
            h_BN3 = self.batch_norm3.feed_forward(h_conv3,is_training,0.99)
            h_dropout3 = tf.nn.dropout(h_BN3,keep_prob)
            h_conv3_post_mask = tf.multiply(h_dropout3,conv3_mask)

        with tf.name_scope('conv4'):
            h_conv4 = tf.nn.relu(conv2d(h_conv3_post_mask, self.W_conv4) + self.b_conv4)
            h_BN4 = self.batch_norm4.feed_forward(h_conv4,is_training,0.99)                        
            h_pool4 = max_pool_2x2(h_BN4)
            h_pool4_post_mask = tf.multiply(h_pool4,conv4_mask)

        # Feature Map Size 8 * 8
        with tf.name_scope('conv5'):
            h_conv5 = tf.nn.relu(conv2d(h_pool4_post_mask, self.W_conv5) + self.b_conv5)
            h_BN5 = self.batch_norm5.feed_forward(h_conv5,is_training,0.99)
            h_dropout5 = tf.nn.dropout(h_BN5,keep_prob)
            h_conv5_post_mask = tf.multiply(h_dropout5,conv5_mask)

        with tf.name_scope('conv6'):
            h_conv6 = tf.nn.relu(conv2d(h_conv5_post_mask, self.W_conv6) + self.b_conv6)
            h_BN6 = self.batch_norm6.feed_forward(h_conv6,is_training,0.99)
            h_dropout6 = tf.nn.dropout(h_BN6,keep_prob)
            h_conv6_post_mask = tf.multiply(h_dropout6,conv6_mask)

        with tf.name_scope('conv7'):
            h_conv7 = tf.nn.relu(conv2d(h_conv6_post_mask, self.W_conv7) + self.b_conv7)
            h_BN7 = self.batch_norm7.feed_forward(h_conv7,is_training,0.99)                                                       
            h_pool7 = max_pool_2x2(h_BN7)
            h_pool7_post_mask = tf.multiply(h_pool7,conv7_mask)

        # Feature Map Size 4 * 4
        with tf.name_scope('conv8'):
            h_conv8 = tf.nn.relu(conv2d(h_pool7_post_mask, self.W_conv8) + self.b_conv8)
            h_BN8 = self.batch_norm8.feed_forward(h_conv8,is_training,0.99)
            h_dropout8 = tf.nn.dropout(h_BN8,keep_prob)            
            h_conv8_post_mask = tf.multiply(h_dropout8,conv8_mask)

        with tf.name_scope('conv9'):
            h_conv9 = tf.nn.relu(conv2d(h_conv8_post_mask, self.W_conv9) + self.b_conv9)
            h_BN9 = self.batch_norm9.feed_forward(h_conv9,is_training,0.99)
            h_dropout9 = tf.nn.dropout(h_BN9,keep_prob)                        
            h_conv9_post_mask = tf.multiply(h_dropout9,conv9_mask)

        with tf.name_scope('conv10'):
            h_conv10 = tf.nn.relu(conv2d(h_conv9_post_mask, self.W_conv10) + self.b_conv10)
            h_BN10 = self.batch_norm10.feed_forward(h_conv10,is_training,0.99)            
            h_pool10 = max_pool_2x2(h_BN10)
            h_pool10_post_mask = tf.multiply(h_pool10,conv10_mask)

        # Feature Map Size 2 * 2
        with tf.name_scope('conv11'):
            h_conv11 = tf.nn.relu(conv2d(h_pool10_post_mask, self.W_conv11) + self.b_conv11)
            h_BN11 = self.batch_norm11.feed_forward(h_conv11,is_training,0.99)  
            h_dropout11 = tf.nn.dropout(h_BN11,keep_prob)                                    
            h_conv11_post_mask = tf.multiply(h_dropout11,conv11_mask)

        with tf.name_scope('conv12'):
            h_conv12 = tf.nn.relu(conv2d(h_conv11_post_mask, self.W_conv12) + self.b_conv12)
            h_BN12 = self.batch_norm12.feed_forward(h_conv12,is_training,0.99)
            h_dropout12 = tf.nn.dropout(h_BN12,keep_prob)                                                
            h_conv12_post_mask = tf.multiply(h_dropout12,conv12_mask)

        with tf.name_scope('conv13'):
            h_conv13 = tf.nn.relu(conv2d(h_conv12_post_mask, self.W_conv13) + self.b_conv13)
            h_BN13 = self.batch_norm13.feed_forward(h_conv13,is_training,0.99)                                    
            h_pool13 = max_pool_2x2(h_BN13)
            h_pool13_post_mask = tf.multiply(h_pool13,conv13_mask)

        # Fully Connected Layers
        with tf.name_scope('fc1'):
            flatten_dim = h_pool13_post_mask.shape[3] # self defined the flatten dimension here 
            h_pool13_flat = tf.reshape(h_pool13_post_mask, [-1, flatten_dim], name = 'Flatten_layer')
            h_flat_droppoed = tf.nn.dropout(h_pool13_flat,keep_prob)                                            
            
            h_fc1 = tf.nn.relu(tf.matmul(h_flat_droppoed, self.W_fc1) + self.b_fc1, name = 'Final_Hidden_Layer')
            h_BN14 = self.batch_norm14.feed_forward(h_fc1,is_training,0.99) 
            
        with tf.name_scope('fc2'):
            h_dropout14 = tf.nn.dropout(h_BN14,keep_prob)
            h_fc1_post_mask = tf.multiply(h_dropout14, afc1_mask)                        
            logits = tf.matmul(h_fc1_post_mask, self.W_fc2) + self.b_fc2
        
        tensor_dict = {'Conv1 Output': h_conv1_post_mask, 'Conv2 Output': h_pool2_post_mask, 
            'Conv3 Output': h_conv3_post_mask, 'Conv4 Output': h_pool4_post_mask, 'Conv5 Output': h_conv5_post_mask, 
            'Conv6 Output': h_conv6_post_mask, 'Conv7 Output': h_pool7_post_mask, 'Conv8 Output': h_conv8_post_mask, 
            'Conv9 Output': h_conv9_post_mask, 'Conv10 Output': h_pool10_post_mask, 'Conv11 Output': h_conv11_post_mask, 
            'Conv12 Output': h_conv12_post_mask, 'Conv13 Output': h_pool13_post_mask, 'Fc1 Output': h_fc1}

        return [tensor_dict, logits]


    def loss(self, logits, labels, ridge):
        labels = tf.to_int64(labels)
        CE_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        fc_l2 = tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_fc2)
        conv_l2 = tf.nn.l2_loss(self.W_conv1) + tf.nn.l2_loss(self.W_conv2) + tf.nn.l2_loss(self.W_conv3) + \
                  tf.nn.l2_loss(self.W_conv4) + tf.nn.l2_loss(self.W_conv5) + tf.nn.l2_loss(self.W_conv6) + \
                  tf.nn.l2_loss(self.W_conv7) + tf.nn.l2_loss(self.W_conv8) + tf.nn.l2_loss(self.W_conv9) + \
                  tf.nn.l2_loss(self.W_conv10) + tf.nn.l2_loss(self.W_conv11) + tf.nn.l2_loss(self.W_conv12) + \
                  tf.nn.l2_loss(self.W_conv13)  

        ridge_loss = ridge * (fc_l2 + conv_l2)
        return [CE_loss,ridge_loss]

    def training(self, opt_name, loss, learning_rate):

        assert opt_name in ['Nestrov_Momentum', 'Adam']
        if opt_name == 'Nestrov_Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)
        elif opt_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        update_op = []
        for bn in self.batch_norm_list:
            update_op.append(bn.moving_mean)
            update_op.append(bn.moving_var)
        update_op.append(train_op)
        # Probably 
        return update_op

    def model_recovery(self, param_list, m_sess):
        # Assigning the weight and bias params
        weight_and_bias = param_list[:len(self.network_param_list)]
        assign_op = []
        for self_param, param in zip(self.network_param_list,weight_and_bias):
            assign_op.append(tf.assign(self_param,param))
        m_sess.run(assign_op)

        # Assigning the batch norm params
        bn_param = param_list[len(self.network_param_list):]
        for self_bn,bn in zip(self.batch_norm_list,bn_param):
            self_bn.bn_recovery(bn,m_sess)

    def model_save(self, m_sess):
        weight_and_bias = m_sess.run(self.network_param_list)
        bn_param = [m_sess.run(bn.param_list) for bn in self.batch_norm_list]
        total_param = weight_and_bias + bn_param
        return total_param
    
    def compressed_model(self, network, mask_list):
        
        # CNN layer
        compressed_net = []
        compressed_net.append(network[0][:,:,:,mask_list[0]])
        compressed_net.append(network[1][mask_list[0]])
        for i in range(1,13):
            compressed_net.append(network[i*2][:,:,mask_list[i-1],:][:,:,:,mask_list[i]]) # Wconv compressed
            compressed_net.append(network[i*2+1][mask_list[i]]) # bconv compressed

        # MLP layer compreesed
        compressed_net.append(network[26][mask_list[12],:][:,mask_list[13]]) # W_fc1
        compressed_net.append(network[27][mask_list[13]])     # b_fc1
        compressed_net.append(network[28][mask_list[13],:])   # W_fc2
        compressed_net.append(network[29])                    # b_fc2

        # Batch Norm
        for layer_i in range(14):
            compressed_net.append([bn_param[mask_list[layer_i]] for bn_param in network[30 + layer_i]])

        return compressed_net

    def print_shape(self):
        shape_info = [int(bias.shape[0]) for bias in self.network_param_list[1:-2:2]]
        print('(Input)3 - ' + str(shape_info) + ' - 10(Output)')
