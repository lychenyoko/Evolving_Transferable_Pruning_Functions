'''
We define the functions that are used in the experiment analysis in this file
'''

def ParseParamKeyWord(param_keyword, file_path):
    '''
    Usage:
        Scan through a file and find the hyper-parameter setting of the experiments
    
    Args:
        param_keyword: (list) of names of hyper-parameters
        file_path: (str) of the file path
    '''
    
    file_read = open(file_path, 'r')
    lines = file_read.readlines()
    
    param_dict = {}
    for line in lines:
        for key in param_keyword:
            if key in line:
                loc = line.find(key) + len(key) + 2
                param_val = line[loc:-1]
                param_dict[key] = param_val              
    
    return param_dict



def GetEpochAcc(epoch_line):
    '''
    Usage:
        Extract the accuracy information of the epoch from a single line of strings
    
    Args:
        epoch_line: (str) of epoch information
    '''
    
    epoch_acc_str = 'Best Acc: '
    ind1 = epoch_line.find(epoch_acc_str, 0)
    ind2 = epoch_line.find(epoch_acc_str, ind1 + len(epoch_acc_str))
    epoch_acc = epoch_line[ind1 + len(epoch_acc_str): ind2 - 1]
    return epoch_acc

def GetCheckAcc(acc_line):
    '''
    Usage:
        Extract the accuracy information of the primary accuracy checking from a single line of strings
    
    Args:
        acc_line: (str) of initial prune step checking information
    '''
    
    str0 = 'is: '
    str1 = '\n'
    ind0 = acc_line.find(str0)
    ind1 = acc_line.find(str1)
    check_acc = acc_line[ind0 + len(str0): ind1]
    return check_acc

def GetFinalPerf(final_perf_line):
    '''
    Usage:
        Extract the accuracy, FLOPs, Parameters information of the
        primary accuracy checking from a single line of strings
    
    Args:
        final_perf_line: (str) of final performance information
    '''
    
    str0 = 'acc_'
    str1 = '_FLOP_'
    str2 = '_Param_'
    str3 = '/ResNet'
    
    ind0 = final_perf_line.find(str0)
    ind1 = final_perf_line.find(str1)
    ind2 = final_perf_line.find(str2)
    ind3 = final_perf_line.find(str3, ind2)
    
    final_acc = final_perf_line[ind0 + len(str0): ind1]
    final_FLOP = final_perf_line[ind1 + len(str1) : ind2]
    final_param = final_perf_line[ind2 + len(str2): ind3]
    
    final_perf_dict = {'Acc': final_acc, 'FLOP': final_FLOP, 'Param': final_param}
    
    return final_perf_dict


def GetPerformance(file_path, is_pruning = True):
    '''
    Usage:
        Get the final performance of a multi-step pruning/training experiments
    
    Args:
        file_path: (str) of the path of the file
        is_pruning: (bool) whether the experiment is pruning or not
    '''
    
    file_read = open(file_path, 'r')
    lines = file_read.readlines()
    
    acc_check_list = []
    epoch_0_line_list = []
    final_perf_list = []
    
    for line in lines:
        if is_pruning:
            if 'Current Accuracy' in line:
                acc_check_list.append(line)
        if 'Epoch #0' in line:
            epoch_0_line_list.append(line)
        if 'file name' in line:
            final_perf_list.append(line)
    
    result_list = []
    
    if is_pruning:
        for acc, epoch0, final_perf in zip(acc_check_list, epoch_0_line_list, final_perf_list):
            result_dict = {}
            result_dict['Check Acc'] = GetCheckAcc(acc)
            result_dict['Epoch 0 Acc'] = GetEpochAcc(epoch0)
            result_dict['Final Performance'] = GetFinalPerf(final_perf)
            result_list.append(result_dict)
    
    else:
        for epoch0, final_perf in zip(epoch_0_line_list, final_perf_list):
            result_dict = {}
            result_dict['Epoch 0 Acc'] = GetEpochAcc(epoch0)
            result_dict['Final Performance'] = GetFinalPerf(final_perf)
            result_list.append(result_dict)
    
    return result_list