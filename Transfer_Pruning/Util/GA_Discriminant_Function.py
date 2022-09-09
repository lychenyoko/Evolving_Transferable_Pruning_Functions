import numpy as np
from .Discriminant_Function import GetPosNegClassStats

def Get_Pos_Stats_and_Vec(X,y):
    '''
    Usage:
        Iterate over different positive class data, obtain the statistics list and the vector list
    
    Return:
        pos_stats_array: (np.array) of dimension [N_class, 2]
        pos_vec_array:   (np.array) of dimension [N_class, data_dim]
    
    '''
    uni_y, count_y = np.unique(y, return_counts=True)
    
    pos_stats_list = []
    pos_vec_list = []
    for label in uni_y:
        pos_class_ind = np.where(y==label)[0]
        pos_X = X[pos_class_ind]
        
        # pos_stats
        pos_stats = np.sum(pos_X), np.sum(np.square(pos_X))
        pos_stats_list.append(pos_stats)
        
        # pos_vec
        pos_vec = np.sum(pos_X, axis=0)
        pos_vec_list.append(pos_vec)
    
    pos_stats_array = np.array(pos_stats_list)
    pos_vec_array = np.array(pos_vec_list)
    
    return pos_stats_array, pos_vec_array, count_y


def Get_Pos_Neg_Mean_Variance(pos_stats_array, data_dim, pos_num, neg_num):
    '''
    Usage:
        Get the statistics overall mean and variance for each positive class and negative class
    
    Return:
        pos_mean_array: (np.array) 1D [N_classes] calculating np.mean(P) of all classes
        neg_mean_array: (np.array) 1D [N_classes] calculating np.mean(Q) of all classes
        
        pos_var_array:  (np.array) 1D [N_classes] calculating np.var(P) of all classes
        neg_var_array:  (np.array) 1D [N_classes] calculating np.var(Q) of all classes        
    '''
    
    total_sum = np.sum(pos_stats_array, axis=0)
    neg_stats_array = total_sum - pos_stats_array # (sum[QX], sum[QX^2])
    
    pos_mean_array = pos_stats_array[:,0] / (pos_num * data_dim) # E[P]
    pos_var_array  = pos_stats_array[:,1] / (pos_num * data_dim) - np.square(pos_mean_array) # E[P^2] - (E[P])^2
    
    neg_mean_array = neg_stats_array[:,0] / (neg_num * data_dim)
    neg_var_array  = neg_stats_array[:,1] / (neg_num * data_dim) - np.square(neg_mean_array)

    return pos_mean_array, neg_mean_array, pos_var_array, neg_var_array


def Get_Pos_Neg_Vector(pos_vec_array, pos_num, neg_num):
    '''
    Usage:
        Get the positive and negative class centroid over all classes
    
    Return:
        pos_vec_mean: (np.array) class centroid of dimension [N_class, data_dim]
        neg_vec_mean: (np.array) class centroid of dimension [N_class, data_dim]
        mX:           (np.array) overall data centroid
    '''
    sumX = np.sum(pos_vec_array, axis=0)
    mX = sumX / np.sum(pos_num)
    
    neg_vec_array = sumX - pos_vec_array
    
    pos_vec_mean = (pos_vec_array.T / pos_num).T
    neg_vec_mean = (neg_vec_array.T / neg_num).T
    
    return pos_vec_mean, neg_vec_mean, mX

def Get_Pos_Neg_Class_Stats_and_Vector(X,y):
    '''
    Usage:
        Base Function to Obtain Statistics that mainly used for self-defined func
        
    Args:
        X: (np.array) of 2D data array w/ [N_samples, N_features]
    
    Returns:
        pos_mean_array: (np.array) 1D [N_classes] calculating np.mean(P) of all classes
        neg_mean_array: (np.array) 1D [N_classes] calculating np.mean(Q) of all classes
    '''
    
    assert len(X.shape) == 2
    data_dim = X.shape[1]

    pos_stats_array, pos_vec_array, count_y = Get_Pos_Stats_and_Vec(X,y)

    pos_num = count_y # NP
    neg_num = np.sum(count_y, axis = 0) - count_y # NQ
    
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array = Get_Pos_Neg_Mean_Variance(pos_stats_array, data_dim, pos_num, neg_num)
    pos_vec_mean, neg_vec_mean, mX = Get_Pos_Neg_Vector(pos_vec_array, pos_num, neg_num)

    return pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num, neg_num, pos_vec_mean, neg_vec_mean, mX



def GA_Func1(X,y):
    '''
    Usage:
        Scalable implementation of the best GA function in population: 
            Co_Evolve_10_gen_40_ind_2020-01-24_18:31:45/population/Gen_5 
    '''
    
    X = X.reshape(X.shape[0], -1) # reshape to 2D vector of dimension [N, D]
    
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num_array, neg_num_array = GetPosNegClassStats(X,y)
    mX = np.mean(X, axis=0)
    
    GA_score_list = []
    for i in range(len(pos_var_array)):
        pos_var, neg_var = pos_var_array[i], neg_var_array[i]
        numerator = np.square(np.subtract(neg_var, mX))
        deno = neg_var + pos_var + 0.05
        score = np.sum(pos_var + numerator / deno)
        GA_score_list.append(score)
    
    return np.sum(GA_score_list)


def GA_Func2(X,y):
    '''
    Usage:
        scalabel implementation of the best individual from the generation:
            Co_Evolve_15_gen_20_ind_2019-12-20_12:29:14/population/Gen_10
    '''
    
    X = X.reshape(X.shape[0], -1) # reshape to 2D vector of dimension [N, D]
    
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num_array, neg_num_array = GetPosNegClassStats(X,y)
    mX = np.mean(X, axis=0)
    
    GA_score_list = []
    for i in range(len(pos_var_array)):
        pos_var, neg_var, neg_mean = pos_var_array[i], neg_var_array[i], neg_mean_array[i]
        first_term = pos_var / (neg_var + 0.05) + neg_var / (pos_var + 0.05)
        second_deno = neg_var + pos_var + 0.05
        second_nume = np.square(np.std(mX) * neg_var * mX + pos_var - neg_mean)
        
        score = np.sum(first_term + second_nume / second_deno)
        GA_score_list.append(score)
    
    return np.sum(GA_score_list)



def GA_inspired_Func1(X, y, ridge=2e-1):
    '''
    Usage:
        Scalable implementation of a inspired function of GA evolution
            Score = std(P)/std(Q) + std(Q)/std(P) + 
                    ((mX - mP)^2 + (mX - mQ)^2)/(std(P) + std(Q))
    
    Args:
        X: (np.array) of 2D [N_samples, N_features]
        y: (np.array) of 1/2D [N_samples]/[N_samples, 1]
    '''
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array,\
     pos_num, neg_num, pos_vec_mean, neg_vec_mean, mX = Get_Pos_Neg_Class_Stats_and_Vector(X, y)
    
    score_list = []
    for i in range(len(pos_var_array)):
        pos_std, neg_std, pos_m, neg_m = \
            np.sqrt(pos_var_array[i]), np.sqrt(neg_var_array[i]), pos_vec_mean[i,:], neg_vec_mean[i,:]
        
        first_term = pos_std / (neg_std + ridge) + neg_std / (pos_std + ridge)
        second_term = (np.sum((mX - pos_m)**2) + np.sum((mX - neg_m)**2)) / (pos_std + neg_std + ridge)
        score = first_term + second_term
        score_list.append(score)
    
    return np.sum(score_list)


def GA_Func3(X,y):
    '''
    Usage:
        scalabel implementation of the best individual from the generation:
            Co_Evolve_10_gen_40_ind_2020-02-04_20:42:06/population/Gen_4
    '''

    X = X.reshape(X.shape[0], -1) # reshape to 1D vector of dimension [N, D]

    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num_array, neg_num_array = GetPosNegClassStats(X,y)
    mX = np.mean(X, axis=0)

    GA_score_list = []
    for i in range(len(pos_mean_array)):
        pos_var, neg_var, pos_num, neg_num = pos_var_array[i], neg_var_array[i], pos_num_array[i], neg_num_array[i]
        nume = (mX - neg_var) ** 2
        deno = np.sqrt(pos_var / pos_num + neg_var / neg_num + 1e-4)

        score = np.sum(nume / deno)
        GA_score_list.append(score)

    return np.sum(GA_score_list)


def GA_Func4(X,y):
    '''
    Usage:
        scalabel implementation of the 3rd best individual from the generation:
            Co_Evolve_10_gen_40_ind_2020-02-04_20:42:06/population/Gen_9
    '''

    X = X.reshape(X.shape[0], -1) # reshape to 1D vector of dimension [N, D]

    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array,\
     pos_num_array, neg_num_array, pos_vec_mean, neg_vec_mean, mX = Get_Pos_Neg_Class_Stats_and_Vector(X, y)

    GA_score_list = []
    for i in range(len(pos_mean_array)):
        pos_var, neg_var, neg_num, pos_mean, neg_mean, neg_vec = pos_var_array[i], neg_var_array[i],\
                neg_num_array[i], pos_mean_array[i], neg_mean_array[i], neg_vec_mean[i]
        term1 = pos_var / (neg_var / neg_num + 0.05)
        term2 = np.abs(pos_mean - neg_mean)

        score = np.sum(term1 + term2 + 0.05 + neg_vec)
        GA_score_list.append(score)

    return np.sum(GA_score_list)

def GA_inspired_Func2(X,y):
    '''
    Usage:
        Scalable implementation of a function inspired by GA_Func1: 
            Score = (mX - mQ)^2 / (var(P) + var(Q))
            
    '''
    
    X = X.reshape(X.shape[0], -1) # reshape to 2D vector of dimension [N, D]
    
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array,\
     pos_num_array, neg_num_array, pos_vec_mean, neg_vec_mean, mX = Get_Pos_Neg_Class_Stats_and_Vector(X, y)
    
    GA_score_list = []
    for i in range(len(pos_var_array)):
        pos_var, neg_var, neg_vec = pos_var_array[i], neg_var_array[i], neg_vec_mean[i]
        numerator = np.square(np.subtract(mX, neg_vec_mean))
        deno = neg_var + pos_var + 0.05
        score = np.sum(numerator / deno)
        GA_score_list.append(score)
    
    return np.sum(GA_score_list)
