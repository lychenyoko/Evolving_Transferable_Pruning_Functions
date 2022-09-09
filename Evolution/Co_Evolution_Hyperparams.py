from GA_Helper.newPrimSet import arg_for_DF
import numpy as np

# Constant
FITNESS_COMBINATION_METHOD = ['weighted_geometric_mean', 'weighted_arithmetic_mean']
FITNESS_KEY_LIST = ['acc','FLOP','Param']
EVALUATION_TIME_LIMIT = 21600 # unit in second, equal to 6 hours

# Initialization
num_per_func = 2
gen_num = 10
init_pop_loaded = False 
init_pop = 'Saved_Generation/pop_of_Co_Evolve_15_gen_20_ind_2019-12-20_12:30:39_gen_-1.pkl'

# Mutation Hyperparams
mutate_prob = 0.75
mutate_min_height = 0
mutate_max_height = 4
mutate_method = 'Full Tree'

# Crossover Hyperparams
need_crossover = True
crossover_prob = 0.75
crossover_method = 'One Point CX'

# Selection Hyperparams
SEL = ['selTour','selBest']
sel_method = SEL[1]
num_sel = 6
sel_tour_size = 3
DF_copy = 0

# Fitness Hyperparams
fitness_combination_method = FITNESS_COMBINATION_METHOD[0]
alpha = 0.5


# Testing Argument for Valid Mutation 

# Feature Maps Operands
from tensorflow import keras
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
Xtr = train_images.reshape(train_images.shape[0],-1)/255
ytr = train_labels

test_arg = arg_for_DF(Xtr[:,100:110],ytr)

# Filter Operands
Nin, Nout = 5,7
H,W = 3, 3
WI = np.random.random((H,W,Nin))
W = np.random.random((H,W,Nin,Nout))

test_arg += [WI, W]
