import numpy as np
import random

import os
import fnmatch
from copy import deepcopy
import pickle as pkl
import time
import datetime

from deap import  base, gp

#------------------------- PrimitiveSet and Initial Populations -------------------------
from GA_Helper.newPrimSet import DFunc_Individual, Single_Tree_Pset
from GA_Helper.newPrimSet import (
    AbsSNR,
    FDR,
    GetSymDiv,
    GetTstats,
    WL1,
    WL2,
    GeoMedian
)
SymDiv = GetSymDiv()
Tstats = GetTstats()
SOAP =[AbsSNR, FDR, SymDiv, Tstats, WL1, WL2, GeoMedian] # State-of-the-art Population
from GA_Helper.Genetic_Op import ind_mutate, ind_mate, my_TourSel, my_BestSel
from GA_Helper.Evolution_Util import LeNet5_find_ind_from_name, VGG16_find_ind_from_name, get_fitness, joint_fitness 
from GA_Helper.Evolution_Util import Evaluate_Ind_LeNet5, Evaluate_Ind_VGG16
from Co_Evolution_Hyperparams import *

#----------------------------------------------------------------------------------------------
#-------------------------------------- Defining Toolbox --------------------------------------
#----------------------------------------------------------------------------------------------
def Get_Current_Readable_Time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def Get_Readable_Timestamp(timestamp):
    '''
    Args:
        timestamp: (float or int)
    '''
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H:%M:%S")


def Get_Mutation_Crossover_Toolbox(min_tree_height, max_tree_height, pset):
    '''
    Usage:
        Return the toolbox for the mutation and crossover process
    '''
    toolbox = base.Toolbox()
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_= min_tree_height, max_= max_tree_height)
    toolbox.register("mutate", gp.mutUniform, expr = toolbox.expr_mut, pset = pset)
    return toolbox


def Get_Initial_Population():
    '''
    Usage:
        Return the initial population to seed the evolution
    '''

    my_pop = []
    if init_pop_loaded:
        my_pop = pkl.load(open(init_pop,'rb')) # init_pop would be a hyperparam in Evo_Hyperparam files
            
    else:
        for i in range(num_per_func):
            for func in SOAP:
                my_pop.append(DFunc_Individual(expr_tree=func, pset=Single_Tree_Pset))

    return my_pop


def Evolution_Directory_Setup(pop_size):

    '''
    Usage:
        Setup the Evolution Experiments
    '''

    exp_time = Get_Current_Readable_Time()
    
    evolution_dir = './GA_Experiment/'+ 'Co_Evolve_' + str(gen_num) + '_gen_' + str(pop_size) +'_ind_'+ exp_time + '/'
    os.mkdir(evolution_dir)
    
    population_dir = evolution_dir + 'population/'
    os.mkdir(population_dir)
    
    result_dir = evolution_dir + 'result/'
    os.mkdir(result_dir)
    
    slurm_dir = evolution_dir + 'slurm/'
    os.mkdir(slurm_dir)
    
    fit_stats_file = evolution_dir + 'fitness_stats_' + exp_time + '.npy'
    hall_of_fame_file =  evolution_dir  + 'best_inds_' + exp_time + '.pkl'
    
    return evolution_dir, population_dir,  result_dir, slurm_dir, fit_stats_file, hall_of_fame_file, exp_time


def Log_Experiment_Status(evolution_dir, pop_size):
    '''
    Usage:
        Print and Log the experiments status
    '''
    Experiment_Log = '------------------------------- GA Overview -------------------------------' + '\n' + '\n' + \
                     'Population:' + '\n' + \
                     '    Num per func: ' + str(num_per_func) + '\n' + \
                     '    Popsize: ' + str(pop_size) + '\n' + \
                     '    Numgen:  ' + str(gen_num) + '\n' + \
                     '    Init Pop Loaded: ' + str(init_pop_loaded) + '\n' + \
                     '    Init Pop: ' + str(init_pop) + '\n' + '\n' + \
                     'Mutation and Crossover:' + '\n' + \
                     '    Mutation Prob: ' + str(mutate_prob) + '\n' + \
                     '    Mutation Min Height: ' + str(mutate_min_height) + '\n' + \
                     '    Mutation Max Height: ' + str(mutate_max_height) + '\n' + \
                     '    Mutation Method: ' + str(mutate_method) + '\n' + \
                     '    Perform Crossover: ' + str(need_crossover) + '\n' + \
                     '    Crossover Prob: ' + str(crossover_prob) + '\n' + \
                     '    Crossover Method: ' + str(crossover_method) + '\n' +  '\n' + \
                     'Selection: ' + '\n' + \
                     '    Select Method: ' + str(sel_method) + '\n' + \
                     '    Num Selection: ' + str(num_sel) + '\n' + \
                     '    Number of DF individuals (/4): ' + str(DF_copy) + '\n' + \
                     '    Tour Size (if Applicable): ' + str(sel_tour_size) + '\n' + \
                     '    Fitness Combination: ' + str(fitness_combination_method) + '\n' + \
                     '    Alpha: ' + str(alpha) + '\n'
    print(Experiment_Log)
    
    Exp_Log_File = open(evolution_dir + 'Experiment_Log.txt','w')
    Exp_Log_File.write(Experiment_Log)
    Exp_Log_File.close()


def Selection_Process(my_pop, pop_size):

    # Selection
    assert(sel_method in ['selTour','selBest'])
    if sel_method == 'selTour':
        parent_indv = my_TourSel(my_pop, sel_tour_size, num_sel)
    elif sel_method == 'selBest':
        parent_indv = my_BestSel(my_pop, num_sel)
    
    assert(len(parent_indv) == num_sel)

    # Segment of D.F. child 
    DF_ind = [DFunc_Individual(expr_tree = tree, pset = Single_Tree_Pset) for tree in SOAP] * DF_copy

    # Segment of reproduced child 
    reproduced_indv = []
    for i in range(pop_size - num_sel - len(DF_ind)):
        indv = deepcopy(random.choice(parent_indv))
        indv.fitness = None
        reproduced_indv.append(indv)

    # Final child individual
    child_indv = DF_ind + reproduced_indv
    random.shuffle(child_indv)

    return parent_indv, child_indv


def Mutation_Crossover_Process(child_indv, toolbox):

    # Mutation and Crossover
    print('\n' + '---------- Mutation and Crossover ----------' + '\n')
    if need_crossover:
        for i in range(0,len(child_indv),2):
            if (random.random() < crossover_prob) and (i + 1 < len(child_indv)):
                ind_mate(child_indv[i],child_indv[i+1],toolbox,test_arg)
    
    for i in range(len(child_indv)):
        if random.random() < mutate_prob:
            ind_mutate(child_indv[i],toolbox,test_arg)

def Dump_Old_Inds(my_pop, Gen_dir):
    '''
    Usage:
        Find out the older individuals and save them
    '''    
    old_inds = [ind for ind in my_pop if ind.fitness is not None]
    if len(old_inds) > 0:
        old_dir = Gen_dir + 'old/'
        os.mkdir(old_dir)
        for i,indv in enumerate(old_inds):
            pkl.dump(indv, open(old_dir + 'ind_' + str(i) + '.pkl', 'wb'))

    return old_inds


def Evaluate_New_Inds(new_inds, Gen_dir, slurm_dir, result_dir, exp_time, g):
    '''
    Usage:
        To save and evaluate the new individuals
    '''
    new_dir = Gen_dir + 'new/'
    os.mkdir(new_dir)
    
    for i,indv in enumerate(new_inds):
        pkl.dump(indv, open(new_dir + 'ind_' + str(i) +'.pkl', 'wb'))    
    
    # Then evaluate each kid and save the evaluation result
    Res_dir = result_dir + 'Gen_' + str(g) + '/'
    os.mkdir(Res_dir)
    print('The results of generation ' + str(g) + ' will be saved in: ' + Res_dir)
    
    print('\n' + '---------- Evaluation of Mutants ----------' + '\n')
    for i in range(len(new_inds)):
        ind_file = new_dir + 'ind_' + str(i) + '.pkl'

        # Evaluate LeNet5
        LeNet5_slurm_out_file = slurm_dir + 'LeNet5_gen_' + str(g) + '_ind_' + str(i) + '.out'
        Evaluate_Ind_LeNet5(LeNet5_slurm_out_file, ind_file, g, i, Res_dir, exp_time, slurm_dir)

        # Evaluate VGG16
        VGG16_slurm_out_file = slurm_dir + 'VGG16_gen_' + str(g) + '_ind_' + str(i) + '.out'
        Evaluate_Ind_VGG16(VGG16_slurm_out_file, ind_file, g, i, Res_dir, exp_time, slurm_dir)

    return Res_dir   


def Check_All_Inds_Finished(Res_dir, slurm_dir, g, new_inds):
    '''
    Usage:
        Check whether all the inidividual evaluations are finished
        
    Args:
        slurm_dir: (str) the directory of slurm jobs output
        Res_dir: (str) the directory of evaluation model output
    '''
    
    All_Inds_Submitted = False
    All_Inds_Finished = False
    
    # First checking loop, check whether all individuals are submitted
    while not All_Inds_Submitted:
        file_list = os.listdir(slurm_dir)
        slurm_out_file = [file for file in file_list if ('_gen_' + str(g) + '_' in file) and ('.out' in file)]
        if len(slurm_out_file) >= 2 * len(new_inds):
            all_submitted_time = time.time()
            all_submitted_time_readable = Get_Readable_Timestamp(all_submitted_time)
            All_Inds_Submitted = True
            print('All individuals are submitted at time: ' + all_submitted_time_readable)
        else:
            time.sleep(10)
    
    # Second checking loop, check whether all individuals are finished
    while not All_Inds_Finished:
        LeNet5_pattern = 'LeNet5_Ind*npy'
        VGG16_pattern = 'VGG16_Ind*npy'

        tmp_list = os.listdir(Res_dir)
        LeNet5_result_list = []
        VGG16_result_list = []

        for entry in tmp_list:
            if fnmatch.fnmatch(entry, LeNet5_pattern):
                 LeNet5_result_list.append(entry)

            if fnmatch.fnmatch(entry, VGG16_pattern):
                 VGG16_result_list.append(entry)

        cur_time = time.time()
        cond1 = (len(LeNet5_result_list) + len(VGG16_result_list)) >= (2 * len(new_inds)) # all individuals finished
        cond2 = (cur_time - all_submitted_time) > EVALUATION_TIME_LIMIT # exceed time limits exception
        if cond1 or cond2:
            cur_time_readable = Get_Readable_Timestamp(cur_time)
            All_Inds_Finished = True
            print('All individuals are finished at time: ' + cur_time_readable)

        else:
            time.sleep(10)

    return LeNet5_result_list, VGG16_result_list


def Get_Generation_Fitnesses(old_inds, new_inds, LeNet5_result_list, VGG16_result_list, print_info = False):
    '''
    Usage:
        Get the fitness values of all the individuals of a generation
        Assign new_indv with fitness
    '''

    # Get the fitness for the old individuals first
    fit_list = [indv.fitness for indv in old_inds]

    # Evaluation Results
    if print_info:
        sorted_LeNet5_result_list = sorted(LeNet5_result_list, key = LeNet5_find_ind_from_name)
        sorted_VGG16_result_list = sorted(VGG16_result_list, key = VGG16_find_ind_from_name)
        print('Result File: ')
        print(str(sorted_LeNet5_result_list) + '\n')
        print(str(sorted_VGG16_result_list) + '\n')

    # Get the fitness for the new individuals
    LeNet5_acc_list = [0] * len(new_inds) # Fitness initialized to 0 (failed, timeout individual 0 fitness)
    for file in LeNet5_result_list:
        ind = LeNet5_find_ind_from_name(file)
        LeNet5_acc_list[ind] = get_fitness(file)['acc']
        
    VGG16_acc_list = [0] * len(new_inds) # Fitness initialized to 0 (failed, timeout individual 0 fitness)
    for file in VGG16_result_list:
        ind = VGG16_find_ind_from_name(file)
        VGG16_acc_list[ind] = get_fitness(file)['acc']
    
    for i,indv in enumerate(new_inds):
        VGG16_acc, LeNet5_acc = VGG16_acc_list[i], LeNet5_acc_list[i]
        fitness = joint_fitness(VGG16_acc, LeNet5_acc, alpha, method = fitness_combination_method)
        indv.fitness = fitness
        indv.VGG16_acc = VGG16_acc
        indv.LeNet5_acc = LeNet5_acc
        fit_list.append(fitness)

    if print_info:
        print('The fitness list: ')
        print(str(fit_list) + '\n')

    return fit_list


def Save_Individuals_and_Fitness_Stats(my_pop, best_individuals, hall_of_fame_file, fitness_stats, fit_stats_file):
    '''
    Usage:
        To save the fitness statistics as well as the best individual of every generation
    '''
 
    the_best = my_BestSel(my_pop, 1)[0] # Selecting Best
    best_individuals.append(the_best)
    pkl.dump(best_individuals, open(hall_of_fame_file, "wb")) # Save the best individual in every generation
    np.save(fit_stats_file, fitness_stats) # Save the fitness statistics
    
    the_best_fit = the_best.fitness
    return the_best_fit

# Main Program
def main():

    # -------------------------------- Setup --------------------------------
    my_pop = Get_Initial_Population() # Get initial population
    pop_size = len(my_pop)
    my_toolbox = Get_Mutation_Crossover_Toolbox(min_tree_height = mutate_min_height, max_tree_height = mutate_max_height, pset = Single_Tree_Pset)
    evolution_dir, population_dir, result_dir, slurm_dir, fit_stats_file, hall_of_fame_file, exp_time = Evolution_Directory_Setup(pop_size)
    Log_Experiment_Status(evolution_dir, pop_size)   

    # -------------------------------- Parallel Evolution --------------------------------
    fitness_stats = []
    best_individuals = []

    for g in range(gen_num):

        # ------------------------- Selection, Mutation, Crossover -------------------------
        all_has_fitness = np.all([ind.fitness is not None for ind in my_pop])   
        if all_has_fitness: # The generation where all individuals have fitness will go through selection and mutation procedure             
            parent_indv, child_indv = Selection_Process(my_pop, pop_size)           
            del my_pop # previous my_pop should be deleted and never showed up in mutation crossover process    
            Mutation_Crossover_Process(child_indv, my_toolbox)
            my_pop = parent_indv + child_indv # Assign bcak the population variable

        # ------------------------- Evaluation -------------------------
            
        # Create Generation Directory
        Gen_dir = population_dir + 'Gen_' + str(g) + '/'
        print('\n' + '------------------------------- Generation ' + str(g) + ' Starts -------------------------------')
        print('The individuals of generation ' + str(g) + ' will be saved in: ' + Gen_dir)
        os.mkdir(Gen_dir)
        
        # Find out the old individuals
        old_inds = Dump_Old_Inds(my_pop, Gen_dir)   
    
        # Find out the new individuals
        new_inds = [ind for ind in my_pop if ind.fitness is None]
        if len(new_inds) > 0:
            Res_dir = Evaluate_New_Inds(new_inds, Gen_dir, slurm_dir, result_dir, exp_time, g)
            
            # The head node will sleep before all individuals are evaluated
            LeNet5_result_list, VGG16_result_list = Check_All_Inds_Finished(Res_dir, slurm_dir, g, new_inds)

        fit_list = Get_Generation_Fitnesses(old_inds, new_inds, LeNet5_result_list, VGG16_result_list, print_info = True)
        my_pop = old_inds + new_inds # Redefine my_pop
        fitness_stats.append(fit_list)
    
        #Saving!
        the_best_fit = Save_Individuals_and_Fitness_Stats(my_pop, best_individuals, hall_of_fame_file, fitness_stats, fit_stats_file)
    
        # Log The Result
        print('\n' + '------------------------------- Generation ' + str(g) + ' Summary' +' -------------------------------')
        print('The best individual has a fitness of: ' + str(the_best_fit) + '\n')

if __name__ == '__main__':
    main()    
