import numpy as np
import tensorflow as tf
import itertools

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

'''True Lenet5 Params'''
conv_ker_size = 5
conv1_map = 20
conv2_map = 50
flatten_dim = 800
flatten_factor = 16
fc_1_dim = 500
output_dim = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
CONV2_SIZE = 4
CONV2_PIXELS = CONV2_SIZE * CONV2_SIZE

def weight_variable(shape,_name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=_name)
    return tf.Variable(initial)

def bias_variable(shape,_name):
    initial = tf.constant(0.0, shape=shape,name=_name)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

class LeNet_5:
   
    def __init__(self, param_list = None, fla_sel = None):
        
        if param_list == None:
            # Convolutional Layer
            self.W_conv1 = weight_variable([conv_ker_size, conv_ker_size, 1, conv1_map], 'W_conv1')
            self.b_conv1 = bias_variable([conv1_map], 'b_conv1')
            self.W_conv2 = weight_variable([conv_ker_size, conv_ker_size, conv1_map, conv2_map], 'W_conv2')
            self.b_conv2 = bias_variable([conv2_map], 'b_conv2')

            # Fully Connected Layer
            self.W_fc1 = weight_variable([flatten_dim, fc_1_dim], 'W_fc1')
            self.b_fc1 = bias_variable([fc_1_dim], 'b_fc1')
            self.W_fc2 = weight_variable([fc_1_dim, output_dim], 'W_fc2')
            self.b_fc2 = bias_variable([output_dim], 'b_fc2')
        
        else:
            # Convolutional Layer
            self.W_conv1 = tf.Variable(param_list[0], name = 'W_conv1')
            self.b_conv1 = tf.Variable(param_list[1], name = 'b_conv1')
            self.W_conv2 = tf.Variable(param_list[2], name = 'W_conv2')
            self.b_conv2 = tf.Variable(param_list[3], name = 'b_conv2')

            # Fully Connected Layer
            self.W_fc1 = tf.Variable(param_list[4], name = 'W_fc1')
            self.b_fc1 = tf.Variable(param_list[5], name = 'b_fc1')
            self.W_fc2 = tf.Variable(param_list[6], name = 'W_fc2')
            self.b_fc2 = tf.Variable(param_list[7], name = 'b_fc2')
        
        if fla_sel == None:
            self.fla_sel = [True] * (self.b_conv2.shape[0] * flatten_factor)
        else:
            self.fla_sel = fla_sel
        
        self.var_list = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                         self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        
    # Initialze Varaibale for the network
    def var_initialization(self, m_sess):
        var_list = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                       self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        for var in var_list:
            m_sess.run(var.initializer)
    
    def get_mask_pl(self):
        conv1_mask = tf.placeholder(tf.float32, shape = [self.b_conv1.shape[0]])
        conv2_mask = tf.placeholder(tf.float32, shape = [self.b_conv2.shape[0]])
        fla_mask   = tf.placeholder(tf.float32, shape = [sum(self.fla_sel)])
        fc1_mask   = tf.placeholder(tf.float32, shape = [self.b_fc1.shape[0]])
        return [conv1_mask,conv2_mask,fla_mask,fc1_mask]
    
    def inference(self,images,conv1_mask=None,conv2_mask=None,fla_mask = None, afc1_mask=None):
        
        # Initialize them for mask free operation
        if conv1_mask == None:
            conv1_mask = [True] * self.b_conv1.shape[0]
        
        if conv2_mask == None:
            conv2_mask = [True] * self.b_conv2.shape[0]

        if fla_mask == None:
            fla_mask = [True] * (sum(self.fla_sel))

        if afc1_mask == None:
            afc1_mask = [True] * self.b_fc1.shape[0]
        
        # building the graph here
        with tf.name_scope('conv1'):
            x_image = tf.reshape(images, [-1, 28, 28, 1])
            h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
            h_pool1_post_mask = tf.multiply(h_pool1,conv1_mask)

        with tf.name_scope('conv2'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1_post_mask, self.W_conv2) + self.b_conv2)
            h_pool2 = max_pool_2x2(h_conv2) 
            h_pool2_post_mask = tf.multiply(h_pool2,conv2_mask)  

        with tf.name_scope('flatten'):
            flatten_dim = h_pool2.shape[1] * h_pool2.shape[2] * h_pool2.shape[3] # self defined the flatten dimension here
            h_pool2_transposed = tf.transpose(h_pool2_post_mask,(0,3,1,2))
            h_pool2_flat = tf.reshape(h_pool2_transposed, [-1, flatten_dim], name = 'Flatten_layer')
            if all(self.fla_sel) == False:
                h_flat = tf.transpose(tf.boolean_mask(tf.transpose(h_pool2_flat),self.fla_sel))
            else:
                h_flat = h_pool2_flat

        with tf.name_scope('fc1'):
            h_flat_post_mask = tf.multiply(h_flat,fla_mask)
            h_fc1 = tf.nn.relu(tf.matmul(h_flat_post_mask, self.W_fc1) + self.b_fc1, name = 'Final_Hidden_Layer')

        with tf.name_scope('fc2'):
            h_fc1_post_mask = tf.multiply(h_fc1,afc1_mask)
            logits = tf.matmul(h_fc1_post_mask, self.W_fc2) + self.b_fc2
        
        tensor_dict = {'Conv1 Output': h_pool1_post_mask, 'Conv2 Output': h_pool2_post_mask, 
                       'Flatten Output': h_flat_post_mask, 'Fc1 Output': h_fc1_post_mask}

        return [tensor_dict, logits]


    def loss(self, logits, labels, ridge):
        labels = tf.to_int64(labels)
        CE_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        ridge_loss = ridge * (tf.nn.l2_loss(self.W_fc1) + tf.nn.l2_loss(self.W_fc2))
        return CE_loss + ridge_loss


    def training(self, opt_name, loss, learning_rate):
        
        assert (opt_name in ['Adam', 'Nestrov Momentum'])
        if opt_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif opt_name == 'Nestrov Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True)
        
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def model_recovery(self, param_list):
        assign_op = [tf.assign(self.W_conv1,param_list[0]), tf.assign(self.b_conv1,param_list[1]),
                     tf.assign(self.W_conv2,param_list[2]), tf.assign(self.b_conv2,param_list[3]),
                     tf.assign(self.W_fc1,param_list[4]), tf.assign(self.b_fc1,param_list[5]),
                     tf.assign(self.W_fc2,param_list[6]), tf.assign(self.b_fc2,param_list[7])]
        return assign_op

    def model_save(self, m_sess):
        retrieve_op = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                       self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        param_in_numpy = m_sess.run(retrieve_op)
        return param_in_numpy
    
    def compressed_model(self, mask_list, m_sess):
        conv1_mask, conv2_mask, fla_mask, fc1_mask = mask_list
        network = self.model_save(m_sess)        
        compressed_network = list(network)
        
        compressed_network[0] = network[0][:,:,:,conv1_mask] # Wconv1
        compressed_network[1] = network[1][conv1_mask]       # bconv1
        
        compressed_network[2] = network[2][:,:,conv1_mask,:][:,:,:,conv2_mask] # Wconv2
        compressed_network[3] = network[3][conv2_mask]                         # bconv2
        
        compressed_network[4] = network[4][fla_mask][:,fc1_mask] # Wfc1
        compressed_network[5] = network[5][fc1_mask]             # bfc1
        compressed_network[6] = network[6][fc1_mask]             # Wfc2
        
        # handle the fla_sel at the end
        comp_fla_sel = np.array(list(self.fla_sel))
        val_fla_sel = np.where(comp_fla_sel == True)[0]        
        comp_fla_sel[val_fla_sel] = np.array(fla_mask)

        dense_dim = list(itertools.chain.from_iterable([[i] * CONV2_PIXELS for i in conv2_mask]))        
        comp_fla_sel = comp_fla_sel[dense_dim]

        compressed_Lenet5 = LeNet_5(param_list = compressed_network, fla_sel = list(comp_fla_sel))
        return compressed_Lenet5
        