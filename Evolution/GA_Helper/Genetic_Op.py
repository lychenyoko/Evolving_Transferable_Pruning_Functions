# Define the two genetic operations: mutation and crossover for evolution

import time
import numpy as np
from deap import gp
import math
SINGLE_EVAL_TIME = 10
import random
from .newPrimSet import Metric_Score

# --------------------------------- Defining Mutation and Mating Process ---------------------------------

# Mutation and Mating Function Just for Expr Tree
def _tree_mutate(ind, tbox, _pset, arg):
    tmp = gp.PrimitiveTree(ind) # First deep copy the individual
    new_ind = gp.PrimitiveTree(ind)    
    while True:
        tbox.mutate(new_ind)
        try:
            mut_func = gp.compile(pset=_pset,expr=new_ind)            
            start_time = time.time()

            m_result = Metric_Score(func = mut_func, arg = arg)

            end_time = time.time()
            time_takes = end_time - start_time
                        
            time_val_flag = time_takes < SINGLE_EVAL_TIME
            result_val_flag = (np.isscalar(m_result)) and (m_result!= np.inf) and \
                       (m_result != -np.inf) and (not math.isnan(m_result))
            val_flag = result_val_flag and time_val_flag
            # Add scalar checking here
            if val_flag:
                print(m_result)
                print('Successfully Mutated!')
                #print(new_ind)
                break
            else:
                if not result_val_flag:
                    print('Not valid value, still need to be mutated!')
                if not time_val_flag:
                    print('Too much time, still need to be mutated!')
                new_ind = gp.PrimitiveTree(tmp)
        except:
            new_ind = gp.PrimitiveTree(tmp)
            print('Exception, still need to be mutated')
    return new_ind


def _tree_mate(ind1, ind2, tbox, _pset, arg):
    tmp1 = gp.PrimitiveTree(ind1) # First deep copy the individual
    new_ind1 = gp.PrimitiveTree(ind1)
    
    tmp2 = gp.PrimitiveTree(ind2)
    new_ind2 = gp.PrimitiveTree(ind2)    
    
    while True:
        tbox.mate(new_ind1,new_ind2)
        try:
            mut_func1 = gp.compile(pset=_pset,expr=new_ind1)
            mut_func2 = gp.compile(pset=_pset,expr=new_ind2)
            
            start = time.time()
 
            m_result1 = Metric_Score(func = mut_func1, arg = arg)
            m_result2 = Metric_Score(func = mut_func2, arg = arg)
            
            end = time.time()
            time_takes = end - start
            
            # Add scalar checking here
            time_val_flag = time_takes < 2 * SINGLE_EVAL_TIME
            flag1 = (np.isscalar(m_result1)) and (m_result1!= np.inf) and \
                    (m_result1 != -np.inf) and (not math.isnan(m_result1))
            flag2 = (np.isscalar(m_result2)) and (m_result2!= np.inf) and \
                    (m_result2 != -np.inf) and (not math.isnan(m_result2))
            
            
            if flag1 and flag2 and time_val_flag:
                print(m_result1)
                print(m_result2)
                print('Successfuly Mated!')
                break
            else:
                if (not flag1) or (not flag2):
                    print('Not valid value, still need to be mated!')
                if not time_val_flag:
                    print('Too much time, still need to be mated!')

                new_ind1 = gp.PrimitiveTree(tmp1)
                new_ind2 = gp.PrimitiveTree(tmp2)                
        except:
            new_ind1 = gp.PrimitiveTree(tmp1)
            new_ind2 = gp.PrimitiveTree(tmp2)
            print('Exception, still need to be mated!')
    return (new_ind1,new_ind2)


def ind_mutate(indv, tbox, arg):
    indv.expr_tree = _tree_mutate(indv.expr_tree, tbox, indv.pset, arg)
    indv.fitness = None

def ind_mate(indv1, indv2, tbox, arg):
    new_tree1, new_tree2 = _tree_mate(indv1.expr_tree, indv2.expr_tree, tbox, indv1.pset, arg)
    indv1.expr_tree = new_tree1
    indv2.expr_tree = new_tree2
    indv1.fitness = None
    indv2.fitness = None



# --------------------------------- Defining Selection Process ---------------------------------

# Tournament Selection
def my_TourSel(pre_sel_pop, tour_size, num_sel):
    new_pop = []
    
    sel_ind = np.random.choice(len(pre_sel_pop) ,num_sel * tour_size, replace=False)
    tour = sel_ind.reshape(num_sel,tour_size)
    
    for i in range(num_sel):
        tournament = [pre_sel_pop[ind] for ind in tour[i]]
        best_ind = sorted(tournament, key = lambda k: k.fitness)[-1]
        new_pop.append(best_ind)
        
    return new_pop

# Best Selection
def my_BestSel(pre_sel_pop, num_sel):
    new_pop = sorted(pre_sel_pop, key = lambda k: k.fitness)[-num_sel:]
    return new_pop 
