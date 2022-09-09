import os
import matplotlib.pyplot as plt
import networkx
from deap import gp
import numpy as np

from .Evolution_Util import get_fitness, LeNet5_find_ind_from_name, VGG16_find_ind_from_name

def Get_Evo_Profile_File_List(evo_dir_list, key_name):
    '''
    Usage:
        Obtain a list of file that can profile the evolution process (e.g. the fitness stats list)
    
    Args:
        evo_dir_list: (list) of evolution directory
        key_name: (str) of the key name that distiguish the file 
    '''

    file_list = []
    for evo_dir in evo_dir_list:
        for file in os.listdir(evo_dir):
            if key_name in file:
                file_list.append(os.path.join(evo_dir, file))
    return file_list


def Get_LeNet5_VGG16_Acc_List(gen_result_dir):
    '''
    Usage:
        Get the acc of LeNet5 and VGG16 in a folder
    '''
    result_file = os.listdir(gen_result_dir)
    LeNet5_file = [file for file in result_file if 'LeNet5' in file]
    VGG16_file = [file for file in result_file if 'VGG16' in file]
    LeNet5_acc_list = [get_fitness(file)['acc'] for file in sorted(LeNet5_file, key = LeNet5_find_ind_from_name)]
    VGG16_acc_list = [get_fitness(file)['acc'] for file in sorted(VGG16_file, key = VGG16_find_ind_from_name)]
    return LeNet5_acc_list, VGG16_acc_list


def Get_LeNet5_VGG16_Acc_Stats(evo_dir_list):
    '''
    Usage:
        Obtain the pruned LeNet5's and VGG16's acc in every generation where new individuals are evluated,
        and store them as a list of list
    
    Args:
        evo_dir_list: (list) of directory for each evolution
    '''
    
    LeNet5_acc_stats = []
    VGG16_acc_stats = []
    for evo_dir in evo_dir_list:
        result_dir = os.path.join(evo_dir, 'result')
        gen_result_list = os.listdir(result_dir)
        sorted_gen_result_list = sorted(gen_result_list, key = lambda gen: float(gen[4:]))
        
        for gen_result in sorted_gen_result_list:
            gen_result_dir = os.path.join(result_dir, gen_result)
            LeNet5_acc_list, VGG16_acc_list = Get_LeNet5_VGG16_Acc_List(gen_result_dir)
            
            if (len(LeNet5_acc_list) > 0) and (len(VGG16_acc_list) > 0):
                LeNet5_acc_stats.append(LeNet5_acc_list)
                VGG16_acc_stats.append(VGG16_acc_list)
    
    return LeNet5_acc_stats, VGG16_acc_stats


def Get_func_index(file):
    '''
    Usage:
        Return index of the file in the list
    '''
    start_ind = 4
    end_ind = file.find('.',start_ind)
    return int(file[start_ind: end_ind])


def Get_Gen_Func_Files(gen_pop_dir):
    '''
    Usage:
        Return a generation of function .pkl file in order
    
    Args:
        gen_pop_dir: (str) the directory of the population files
    '''

    old_dir = os.path.join(gen_pop_dir, 'old')
    if os.path.exists(old_dir):
        sorted_old_funcs = sorted(os.listdir(old_dir), key = Get_func_index)
        old_files = [os.path.join(old_dir, file) for file in sorted_old_funcs]
    else:
        print('No old directories!')
        old_files = []

    new_dir = os.path.join(gen_pop_dir, 'new')
    sorted_new_funcs = sorted(os.listdir(new_dir), key = Get_func_index)
    new_files = [os.path.join(new_dir, file) for file in sorted_new_funcs]

    gen_func_files = old_files + new_files 
    return gen_func_files

def Get_Pop_Dir_List(evo_dir_list):
    '''
    Usage:
        Get the population directory of a evolution process in order 
        
    Args:
        evo_dir_list: (list) of successive evolution directory
    '''
    
    pop_dir_list = []
    for evo_dir in evo_dir_list:
        pop_dir = os.path.join(evo_dir, 'population')
        gen_pop_list = os.listdir(pop_dir)
        sorted_gen_pop_list = sorted(gen_pop_list, key = lambda gen: float(gen[4:]))
        pop_dir_list += [os.path.join(pop_dir, gen_pop) for gen_pop in sorted_gen_pop_list]
        
    return pop_dir_list

def PlotIndividual(ind,fig_size):
    '''
    Usage:
        Visualize the expression tree in the format of a networkx graph
    '''

    nodes, edges, labels = gp.graph(ind)
    graph = networkx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos = networkx.nx_pydot.graphviz_layout(graph, prog = 'dot')

    plt.figure(figsize=fig_size)
    networkx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="y")
    networkx.draw_networkx_edges(graph, pos)
    networkx.draw_networkx_labels(graph, pos, labels)
    plt.axis("off")
    plt.show()

def Get_Top_Indices(mlist, k):
    '''
    Usage:
        A helper fucntion to get the indices of top k item in a list 
        
    Args:
        mlist: (list) of comparable number
        k: the top k indices
    '''
    
    indices = sorted(range(len(mlist)), key=lambda i: mlist[i])[-k:]
    return indices


def Get_Sort_Index(score_list):
    '''
    Usage:
        Return a list sort_index with same dimension as score_list where 
        sort_index[i] = rank of score_list[i] in the whole list
    
    Args:
        score_list: (1D list) with score corresponding to each elements
        
    '''
    func_sort = np.argsort(score_list)
    sort_index = np.zeros(len(score_list))
    for i in range(len(score_list)):
        sort_index[func_sort[i]] = i
    return sort_index