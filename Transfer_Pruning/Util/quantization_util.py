import numpy as np
from copy import deepcopy
import itertools


def Kernel_Quantization(kernel, num_bits):
    '''
    Usage:
        To achieve a fake quantization for a convolutional kernel
    
    Input:
        kernel: (numpy.array) 3D array [width, height, out_channel]
        num_bits: (int) number of bits to represent such kernels
    
    Return:
        quan_ker: (numpy.array) quantized 3D array [width, height, out_channel]
    
    '''
    
    quant_levels = np.power(2, num_bits) - 1
    max_value, min_value = np.max(kernel), np.min(kernel)
    
    scale = (max_value - min_value) / quant_levels
    
    if scale == 0:
        quant_index = np.zeros(kernel.shape)
        quant_ker = kernel
    else:
        quant_index = np.round((kernel - min_value) / scale)
        quant_ker =  quant_index * scale + min_value
    
    return quant_ker, quant_index



def Layer_Quantization(kernels, num_bits_list):
    '''
    Usage:
        To achieve a fake quantization for the whole layer
    
    Input:
        kernels: (numpy.array) 4D array [width, height, in_channel, out_channel]
        num_bit_list: (list of int) list of num_bits
    
    Return:
        quan_kers: (numpy.array) quantized 4D array [width, height, in_channel, out_channel]

    '''
    
    quant_kers = np.zeros(kernels.shape).astype(np.float32)
    quant_indices = np.zeros(kernels.shape).astype(np.int)
    assert len(num_bits_list) == kernels.shape[2] # the number of channels should match
    
    for i in range(len(num_bits_list)):
        ker = kernels[:,:,i,:]
        num_bits = num_bits_list[i]
        quant_ker, quant_index = Kernel_Quantization(kernel=ker,num_bits=num_bits)
        quant_kers[:,:,i,:] = quant_ker
        quant_indices[:,:,i,:] = quant_index
    
    return quant_kers, quant_indices



def Net_Quantization(kernels_list, net_num_bits_list):
    '''
    Usage:
        To achieve a fake quantization for the whole network model
    
    Input:
        kernels_list: (list of numpy.array) list of 4D array [width, height, in_channel, out_channel]
        num_bit_list: (list of list of int) list of num_bits_list
    
    Return:
        quan_kers_list: (numpy.array) quantized 4D array [width, height, in_channel, out_channel]

    '''
    
    quant_kers_list = []
    quant_indices_list = []
    assert len(kernels_list) == len(net_num_bits_list) # the number of layers should match
    
    for i in range(len(kernels_list)):
        kernels = kernels_list[i]
        num_bits_list = net_num_bits_list[i]
        quant_kers, quant_indices = Layer_Quantization(kernels=kernels, num_bits_list=num_bits_list)
        quant_kers_list.append(quant_kers)
        quant_indices_list.append(quant_indices)
    
    return quant_kers_list, quant_indices_list


def Generate_Net_Num_Bits_List(Net_Channel_Score, thres, quant_bits):
    '''
    Usage:
        Generate the num_bits_list for the whole network model

    Args:
        Net_Channel_Score: (list of list of float) the metric score for each channel
        thres: (list of float) the thresholds to decide the node quantization level 
               (sorted in desending order)
        quant_bits: (list of int) the set of quantization levels
                    (sorted in descending order)
    
    Return:
        net_num_bits_list: (list of list of int) the quantization bits of each channel
    '''
    
    assert len(thres) == len(quant_bits) - 1
    
    net_num_bits_list = []
    for layer_score in Net_Channel_Score:
        num_bits_list = []
        for node_score in layer_score:
            num_bit = Quant_Bit_from_Thres(node_score, thres, quant_bits)
            num_bits_list.append(num_bit)
        net_num_bits_list.append(num_bits_list)
    
    return net_num_bits_list

def Quant_Bit_from_Thres(node_score, thres, quant_bits):
    '''
    Usage:
        Generate the num_bit for a single node

    Args:
        node_score: (float) the metric score of the channel
        thres: (list of float) the thresholds to decide the node quantization level
        quant_bits: (list of int) the set of quantization levels
    
    Return:
        num_bit: (int) the quantization bit of the channel
    '''    
    if node_score > thres[0]:
        num_bit = quant_bits[0]
    elif node_score <= thres[-1]:
        num_bit = quant_bits[-1]
    else:
        for i in range(0, len(thres) - 1):
            if (node_score <= thres[i]) and (node_score > thres[i + 1]):
                num_bit = quant_bits[i + 1]
                break
    
    return num_bit


def Get_Score_by_Percentile(model_score, percentile):
    '''
    Usage:
        To obtain the score at the percentile of overall model score for 2-precision quantization
    
    Args:
        model_score: (list) of (list) of channel scores in a layer and across all layers
        percentile: (float) between 0 to 1, the score below such percentile will be quantized to low precisions,
                            otherwise, it would be with high precisions.
    '''
    flatten_model_score = list(itertools.chain.from_iterable(model_score))
    sorted_score = sorted(flatten_model_score)
    elements_to_look = int(len(sorted_score) * percentile)
    thres_score = sorted_score[elements_to_look]
    return thres_score


def Get_Quant_Model(model, quant_kers_list, num_ker_per_block):
    '''
    Usage:
        Transform the original model to a quantized model
    
    Args:
        model: (list of list of np.array) the parameters defining a ResNet model
        quant_kers_list: (list of np.array) the quantized kernels parameters
        num_ker_per_block: (int) the number of weight kernels in a block
    
    Return:
        quant_model: (list of list of np.array) quantized parameters of a new ResNet model
    '''
    
    quant_model = deepcopy(model)

    ker_iter = 0
    for ind in range(1, len(quant_model) - 1):
        for i in range(num_ker_per_block):
            quant_model[ind][i] = quant_kers_list[ker_iter + i]
        ker_iter += num_ker_per_block
        
    return quant_model


def Get_Index_Dist(quant_indices_list, net_num_bits_list, precision_bit):
    '''
    Usage:
        Get all the quantization indices of the weights at a particular quantization precision
    Args:
        quant_indices_list: (list) of 4D kernel [width, height, in_channel, out_channel] with quantization index 
        net_num_bits_list: (list) of quantization bits for node
        precision_bit: (int) specified quantization precision
    '''
    
    assert(len(quant_indices_list) == len(net_num_bits_list))
    
    index_list = []
    for layer_ind in range(len(quant_indices_list)):
        quant_indices = quant_indices_list[layer_ind]
        num_bits_list = net_num_bits_list[layer_ind]
        
        for ker_ind in range(len(num_bits_list)):
            quant_index = quant_indices[:,:,ker_ind,:]
            num_bits = num_bits_list[ker_ind]
            
            if num_bits == precision_bit:
                index_list += list(quant_index.reshape(-1))
    
    return index_list


class My_PrioirtyQueue:
    def __init__(self):
        self.mqueue = []
    
    def put(self, item):
        '''
        Usage:
            Add item to the queue, and sort the queue once again
        
        Args:
            item: tuple of the form (frequency, index)
        '''
        self.mqueue.append(item)
        self.mqueue = sorted(self.mqueue, key = lambda pair: pair[0])
        
    def get(self):
        '''
        Usage:
            return the item with the least frequency in self.mqueue
        '''
        pop_item = self.mqueue.pop(0)
        return pop_item
    
    def qsize(self):
        '''
        Usage:
            return the size of the queue
        '''
        return len(self.mqueue)
    
class HuffmanNode:
    '''
    Usage:
        Simple TreeNode definition
    '''
    def __init__(self, left=None, right=None, root=None):
        self.left = left
        self.right = right
        self.root = root     
    def children(self):
        return((self.left, self.right))

def Create_Huffman_Tree(index_list):
    '''
    Usage:
        Build a Huffman Tree and return the root node
    
    Args:
        index_list: (list) of quantization level/index for the weights

    '''
    ind_, ind_counts = np.unique(index_list, return_counts=True)
    ind_freq = ind_counts / sum(ind_counts)
    freq_index_pair = list(zip(ind_freq, ind_))
    
    p = My_PrioirtyQueue()
    for value in freq_index_pair:    # 1. Create a leaf node for each symbol
        p.put(value)                 #    and add it to the priority queue
    while p.qsize() > 1:             # 2. While there is more than one node
        l, r = p.get(), p.get()      # 2a. remove two highest nodes
        node = HuffmanNode(l, r)     # 2b. create internal node with children
        p.put((l[0]+r[0], node))     # 2c. add new node to queue
    return p.get()                   # 3. tree is complete - return root node


def Get_Code(node, prefix="", code={}):
    '''
    Usage:
        Recursively walk the tree down to the leaves, assigning a code value to each symbol
    
    Input:
        node: (tuple) of the form (frequency, HuffmanNode/symbol)
    '''
    if isinstance(node[1].left[1], HuffmanNode):
        Get_Code(node[1].left,prefix+"0", code)
    else:
        code[node[1].left[1]] = prefix+"0"
    if isinstance(node[1].right[1],HuffmanNode):
        Get_Code(node[1].right,prefix+"1", code)
    else:
        code[node[1].right[1]]=prefix+"1"
    return(code)

def Get_Num_Bit_with_Codebook(index_list, codebook):
    '''
    Usage:
        Return the total number of bits needed to encode the a list of quantization level
        
    Args:
        index_list: (list) of quantization level/index for each weight
        codebook: (dict) with key -> quantization index, val -> code
    '''
    ind_, ind_counts = np.unique(index_list, return_counts=True)
    num_bit = 0
    for ind, ind_count in zip(ind_, ind_counts):
        code_len = len(codebook[ind])
        num_bit += code_len * ind_count
    
    return num_bit

def Get_Codebook_Size(codebook):
    '''
    Args:
        Return the size of the codebook in bits
    '''
    code_bit = sum([len(val) for val in list(codebook.values())])
    
    num_symbol = len(list(codebook.values()))
    symbol_bit = num_symbol * np.log2(num_symbol)
    return symbol_bit + code_bit


def Transform_Block_to_fake_16_bit(block_param):
    '''
    Usage:
        To convert the whole block's convolution parameters to 16-bit
    '''
    new_blk_param = deepcopy(block_param)
    for i in range(len(block_param)):
        if len(np.array(block_param[i]).shape) == 4: # check if the parameter is a convolution parameter
            param_fake_16_bit = (block_param[i].astype(np.float16)).astype(np.float32)
            new_blk_param[i] = param_fake_16_bit
    
    return new_blk_param