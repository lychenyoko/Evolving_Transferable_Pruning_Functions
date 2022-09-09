import numpy as np
import itertools
chain_list = lambda c: list(itertools.chain.from_iterable(c))
from Util.mask_util import Print_Mask_Shape

#====================================== ResNet Parameter Calculator ======================================
def ResNetParamCal(ResNet_Object):
    init_param = ResNet_Object.W_conv0.shape
    init_num = init_param[0] * init_param[1] * init_param[2] * init_param[3]
    final_param = ResNet_Object.W_fc0.shape
    final_num = final_param[0] * final_param[1]

    total_blk_param = 0
    for blk in chain_list(ResNet_Object.block_list):
        blk_param = _BlkParamCal(blk)
        total_blk_param = total_blk_param + blk_param
    
    return int(init_num + total_blk_param + final_num)

def _BlkParamCal(blk_obj):
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    weight_param = [shape[0] * shape[1] * shape[2] * shape[3] for shape in blk_shape]
    bn_param = [shape[2] for shape in blk_shape]
    blk_param = sum(weight_param) + sum(bn_param) * 4
    return blk_param



# ====================================== ResNet Bits Calculator ======================================

def ResNetStorageBitCal(ResNet_Object, net_num_bits_list):
    
    '''
    Usage:
        Calculate the number of bits to store for a network
        
    Args:
        ResNet_Object: a ResNet Object
        net_num_bit_lists: 
    '''
    
    init_param = ResNet_Object.W_conv0.shape
    init_bit = np.prod(init_param) * 32 # initial layer is not quantized
    final_param = ResNet_Object.W_fc0.shape
    final_bit = final_param[0] * final_param[1] * 32 # final fc layer is not quantized

    # Check the block type
    if ResNet_Object.block_type == 'Normal':
        kernels_per_block = 2
    else:
        kernels_per_block = 3
    
    total_blk_bit = 0
    for ind, blk in enumerate(chain_list(ResNet_Object.block_list)):
        blk_bit = _BlkStorageBitCal(blk, net_num_bits_list[ind * kernels_per_block : (ind + 1) * kernels_per_block])
        total_blk_bit += blk_bit
    
    return int(init_bit + total_blk_bit + final_bit)

def _BlkStorageBitCal(blk_obj, blk_num_bits_list):
    '''
    Usage:
        Calculate the number of bits within a block, ignore the batch norm for now
    
    Args:
        blk_obj: A Residual Block / BottleNeck Object
        blk_num_bit_lists
    '''
    
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    
    blk_bits = 0
    for i in range(len(blk_shape)):
        weight_bits = np.prod(blk_shape[i]) * np.sum(blk_num_bits_list[i]) // len(blk_num_bits_list[i])
        blk_bits += weight_bits
    return blk_bits




#====================================== ResNet FLOPs Calculator ======================================
def _ResBlockFLOPCal(blk_obj,size_in=None,size_out=None):
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    weight_param = [shape[0] * shape[1] * shape[2] * shape[3] for shape in blk_shape]

    FLOP_list = [weight_param[i] * np.power(size_out, 2) for i in range(2)]
    total_blk_FLOP = sum(FLOP_list)
    return total_blk_FLOP

def _BottleNeckFLOPCal(blk_obj,size_in,size_out):
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    weight_param = [shape[0] * shape[1] * shape[2] * shape[3] for shape in blk_shape]
    if blk_obj.downsample:
        size_list = [size_in,  size_out, size_out]
    else:
        size_list = [size_out, size_out, size_out]
    FLOP_list = [weight_param[i] * np.power(size_list[i], 2) for i in range(3)]
    blk_FLOP = sum(FLOP_list)
    return blk_FLOP

def ResNetFLOPCal(ResNet_Object):
    
    if ResNet_Object.block_type == 'BottleNeck':
        BLK_Cal = _BottleNeckFLOPCal
    if ResNet_Object.block_type == 'Normal':
        BLK_Cal = _ResBlockFLOPCal
    
    # FLOPs of first conv layer
    init_param = ResNet_Object.W_conv0.shape
    init_num = init_param[0] * init_param[1] * init_param[2] * init_param[3]
    init_FLOP = int(init_num) * 32 * 32
    
    # FLOPs of last fc layer
    final_param = ResNet_Object.W_fc0.shape
    final_num = final_param[0] * final_param[1]
    final_FLOP = int(final_num)
    
    # FLOPs from the residual blocks
    total_blk_FLOP = 0
    
    size_in =  [32, 32, 16]
    size_out = [32, 16, 8]
    
    # The FLOPs calculating logic here
    for stage,blk_stage in enumerate(ResNet_Object.block_list):
        stage_size_in, stage_size_out = size_in[stage], size_out[stage]
        
        for blk in blk_stage:
            blk_FLOP = BLK_Cal(blk,size_in = stage_size_in, size_out = stage_size_out)
            total_blk_FLOP = total_blk_FLOP + blk_FLOP
    
    return int(init_FLOP + total_blk_FLOP + final_FLOP)


# ====================================== ResNet Neurons Calculator ======================================
def _ResBlockNeuronCal(blk_obj,size_in=None,size_out=None):
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    weight_param = [shape[3] for shape in blk_shape]

    Neuron_list = [weight_param[i] * np.power(size_out, 2) for i in range(2)]
    total_blk_Neuron = sum(Neuron_list)
    return total_blk_Neuron

def _BottleNeckNeuronCal(blk_obj,size_in,size_out):
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    weight_param = [shape[3] for shape in blk_shape]
    if blk_obj.downsample:
        size_list = [size_in,  size_out, size_out]
    else:
        size_list = [size_out, size_out, size_out]
    Neuron_list = [weight_param[i] * np.power(size_list[i], 2) for i in range(3)]
    blk_Neuron = sum(Neuron_list)
    return blk_Neuron


def ResNetNeuronCal(ResNet_Object):
    
    if ResNet_Object.block_type == 'BottleNeck':
        BLK_Cal = _BottleNeckNeuronCal
    if ResNet_Object.block_type == 'Normal':
        BLK_Cal = _ResBlockNeuronCal
    
    # neurons of first conv layer
    input_neuron = 3 * 32 * 32
    init_neuron = int(ResNet_Object.W_conv0.shape[-1]) * 32 * 32
    
    # neurons of last fc layer
    final_neuron = ResNet_Object.W_fc0.shape[0] + ResNet_Object.W_fc0.shape[1]
    
    # neurons from the residual blocks
    total_blk_neuron = 0
    
    size_in =  [32, 32, 16]
    size_out = [32, 16, 8]

    # The Neurons calculating logic here
    for stage, blk_stage in enumerate(ResNet_Object.block_list):
        stage_size_in, stage_size_out = size_in[stage], size_out[stage]

        for blk in blk_stage:
            blk_Neuron = BLK_Cal(blk, size_in = stage_size_in, size_out = stage_size_out)
            total_blk_neuron = total_blk_neuron + blk_Neuron

    
    return int(init_neuron + total_blk_neuron + final_neuron)



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Numpy FLOP Calculators >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def _ResBlockFLOPCal_NumPy(blk, size_out):
    weight_param = blk[:2]
    weight_num_param = [np.prod(weight.shape) for weight in weight_param]
    FLOP_list = [weight_num_param[i] * np.power(size_out, 2) for i in range(2)]
    total_blk_FLOP = sum(FLOP_list)
    return total_blk_FLOP

def ResNetFLOPCal_Numpy(ResNet_Param_NumPy, num_res_block):

    first_layer_param, blk_param, final_layer_param = ResNet_Param_NumPy[0], ResNet_Param_NumPy[1:-1], ResNet_Param_NumPy[-1]
    # FLOPs of first convolutional layer
    init_FLOP = np.prod(first_layer_param[0].shape) * 32 * 32

    # FLOPs of last fc layer
    final_FLOP = np.prod(final_layer_param[0].shape)
    
    # FLOP from the residual block
    size_out = [32, 16, 8]
    total_blk_FLOP = 0
    
    blk_iter_ind = 0
    for size in size_out:
        for i in range(num_res_block):
            blk = blk_param[blk_iter_ind]
            blk_FLOP = _ResBlockFLOPCal_NumPy(blk, size)
            total_blk_FLOP = total_blk_FLOP + blk_FLOP
            blk_iter_ind = blk_iter_ind + 1
    
    return int(init_FLOP + total_blk_FLOP + final_FLOP)


def Get_FLOP_Reduction_Per_Layer(mask_list, ResNet_Obj, sess, ori_FLOP, info_print = False):
    '''
    Usage:
        Get the network FLOPs reduction for each layer if we remove one channel from it.
    
    Args:
        mask_list: the current mask indicating the network's shape
        ResNet_Obj: the ResNet object to perform the compressed_model method
        sess: The session to run the operation
        ori_FLOP: the current FLOP of the (masked) network
        info_print: Whether or not to print out the information
    
    Return:
        A list showing the reduced FLOP for each layer of the network
    '''
    from copy import deepcopy

    net_reduced_FLOP = []
    for stage_i in range(len(mask_list)):
        stage_reduced_FLOP = []

        for blk_j in range(len(mask_list[stage_i])):
            blk_reduced_FLOP = []

            for lay_k in range(len(mask_list[stage_i][blk_j])):
                tmp_mask_list = deepcopy(mask_list)
                mask = tmp_mask_list[stage_i][blk_j][lay_k]
                active_node = np.where(mask == True)[0]
                if len(active_node) >= len(mask) * 0.125: # Self defined threshold
                    sel_node = active_node[0]
                    mask[sel_node] = False

                comp_param = ResNet_Obj.compressed_model(mask_list=tmp_mask_list,m_sess=sess)
                reduced_FLOP = ori_FLOP - ResNetFLOPCal_Numpy(comp_param, num_res_block=ResNet_Obj.num_res_blocks)

                if info_print:
                    print('Stage: ' + str(stage_i) + ' Block: ' + str(blk_j) + ' Layer: ' + str(lay_k))
                    Print_Mask_Shape(tmp_mask_list)
                    print('The compressed FLOP is: ' + str(reduced_FLOP))
                    print('')

                blk_reduced_FLOP.append(reduced_FLOP)
            stage_reduced_FLOP.append(blk_reduced_FLOP)
        net_reduced_FLOP.append(stage_reduced_FLOP)
    return net_reduced_FLOP


# A Network Shape Printer
def Print_Network(network_obj):
    net_shape = []
    for stage_i in network_obj.block_list:
        stage_shape = []
        for blk_j in range(len(stage_i)):
            blk = stage_i[blk_j]
            num_per_block = [int(W.shape[-2]) for W in blk.weight_list]
            stage_shape.append(num_per_block)
        net_shape.append(stage_shape)

    for blk in net_shape:
        print(blk)

        
        
#====================================== ResNet Process Bit Calculator ======================================
def _ResBlockProcessBitCal(blk_obj, size_in, size_out, blk_num_bits_list):
    '''
    Usage:
        Calculate the number of bits processed by an ALU when inferencing an image for a normal Residual Block
    
    Args:
        blk_obj:           (Block) object of self-coded block
        size_in:           (int) the spatial size of the input channel of the block
        size_out:          (int) the spatial size of the output channel of the block 
        blk_num_bits_list: (list) of num_bits_list where each element indicate the number of bits for the weights 
    '''
    
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    
    blk_process_bits = 0
    for i in range(len(blk_shape)):
        assert len(blk_num_bits_list[i]) == blk_shape[i][2] # asserting matching of input channels
        lay_FLOP = np.prod(blk_shape[i]) * np.power(size_out, 2)
        lay_processed_bits = (lay_FLOP // len(blk_num_bits_list[i])) * np.sum(blk_num_bits_list[i])
        blk_process_bits += lay_processed_bits
    
    return blk_process_bits

def _BottleNeckProcessBitCal(blk_obj, size_in, size_out, blk_num_bits_list):
    '''
    Usage:
        Calculate the number of bits processed by an ALU when inferencing an image for a BottleNeck Block
    
    Args:
        blk_obj:           (Block) object of self-coded block
        size_in:           (int) the spatial size of the input channel of the block
        size_out:          (int) the spatial size of the output channel of the block    
        blk_num_bits_list: (list) of num_bits_list where each element indicate the number of bits for the weights 
    '''    
    
    blk_shape = [weight.shape for weight in blk_obj.weight_list]
    
    if blk_obj.downsample:
        size_list = [size_in,  size_out, size_out]
    else:
        size_list = [size_out, size_out, size_out]
        
    blk_process_bits = 0
    for i in range(len(blk_shape)):
        assert len(blk_num_bits_list[i]) == blk_shape[i][2] # asserting matching of input channels
        lay_FLOP = np.prod(blk_shape[i]) * np.power(size_list[i], 2)
        lay_processed_bits = (lay_FLOP // len(blk_num_bits_list[i])) * np.sum(blk_num_bits_list[i])
        blk_process_bits += lay_processed_bits
    
    return blk_process_bits

def ResNetProcessBitCal(ResNet_Object, net_num_bits_list):
    '''
    Usage:
        Calculate the number of process bits for inferencing an image with the network
    
    Args:
    
    '''
    NUM_BIT_FOR_CONV0_FC = 32
    SPATIAL_SIZE_CONV0 = 32
    
    if ResNet_Object.block_type == 'BottleNeck':
        kernels_per_block = 3
        BLK_Cal = _BottleNeckProcessBitCal
    if ResNet_Object.block_type == 'Normal':
        kernels_per_block = 2
        BLK_Cal = _ResBlockProcessBitCal
    
    # Processd bits of first conv layer
    conv0_FLOP = np.prod(ResNet_Object.W_conv0.shape) * (SPATIAL_SIZE_CONV0 ** 2)
    conv0_process_bit = conv0_FLOP * NUM_BIT_FOR_CONV0_FC
    
    # Processd bits of last fc layer
    fc_FLOP = np.prod(ResNet_Object.W_fc0.shape)
    fc_process_bit = fc_FLOP * NUM_BIT_FOR_CONV0_FC
    
    # Processd bits from the residual blocks
    size_in =  [32, 32, 16]
    size_out = [32, 16, 8]
    
    total_blk_process_bits = 0
    blk_ind = 0
    for stage,blk_stage in enumerate(ResNet_Object.block_list):
        stage_size_in, stage_size_out = size_in[stage], size_out[stage]
        
        for blk in blk_stage:
            blk_num_bits_list = net_num_bits_list[blk_ind * kernels_per_block: (blk_ind + 1) * kernels_per_block]
            blk_process_bits = BLK_Cal(blk, stage_size_in, stage_size_out, blk_num_bits_list)
            total_blk_process_bits += blk_process_bits
            blk_ind += 1
    
    return int(conv0_process_bit + total_blk_process_bits + fc_process_bit)        