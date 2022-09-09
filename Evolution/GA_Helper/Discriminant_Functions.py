import numpy as np

DI_rho = 1e-4

'''DI Function'''
def GetScatterMatrix(X):
    # input will have row as data

    if len(X.shape) == 1:
        X = X.reshape(-1,1)
    centered_X = X - np.mean(X,axis=0).reshape(1,-1)
    S = centered_X.T.dot(centered_X)
    return S

def GetDeltaMatrix(X,y):
    # input will have row as data
    if len(X.shape) == 1:
        X = X.reshape(-1,1)

    d_mean = np.mean(X,axis=0) # The mean of all the data
    unique_l = np.unique(y)
    DeltaMatrix = np.zeros((X.shape[1],len(unique_l)))
    
    for index,label in enumerate(unique_l):
        mask = np.where(y==label)[0]
        class_mean = np.mean(X[mask,:],axis=0) # Mean within class
        DeltaMatrix[:,index] = class_mean - d_mean
    
    # output will have column as the difference between class mean and data mean 
    return DeltaMatrix

def GetDI(X,y,rho=1e-4):

    SS = GetScatterMatrix(X)
    dim = SS.shape[0]
    D = GetDeltaMatrix(X,y)
    scaled_D = np.zeros(D.shape)
    for index,label in enumerate(np.unique(y)):
        n_y_sample = len(np.where(y==label)[0])
        scaled_D[:,index] = D[:,index] * np.sqrt(n_y_sample)
    tmp = ( np.linalg.inv( SS + rho*np.eye(dim) ) ).dot(scaled_D)
    DI = np.trace(tmp.dot(scaled_D.T))

    return DI


'''
Base Function to Obtain Statistics that Will be Used for FDR, Abs_SNR, SymDiv, TwoT
'''
def GetPosNegClassStats(X,y):
    '''
    Input X must be a 2D array
    '''
    assert len(X.shape) == 2
    data_dim = X.shape[1]
    uni_y, count = np.unique(y,return_counts=True)
    pos_stats_list = []
    for label in uni_y:
        pos_class_ind = np.where(y==label)[0]
        pos_X = X[pos_class_ind]
        pos_stats = np.sum(pos_X), np.sum(np.square(pos_X))
        pos_stats_list.append(pos_stats)
    
    pos_stats_array = np.array(pos_stats_list) # (sum[PX], sum[PX^2])
    total_sum = np.sum(pos_stats_array, axis=0)
    neg_stats_array = total_sum - pos_stats_array # (sum[QX], sum[QX^2])
    
    pos_num = count
    neg_num = np.sum(count, axis = 0) - count # N
    
    pos_mean_array = pos_stats_array[:,0] / (pos_num * data_dim) # E[X]
    pos_var_array  = pos_stats_array[:,1] / (pos_num * data_dim) - np.square(pos_mean_array) # E[X^2] - (E[X])^2
    
    neg_mean_array = neg_stats_array[:,0] / (neg_num * data_dim)
    neg_var_array  = neg_stats_array[:,1] / (neg_num * data_dim) - np.square(neg_mean_array)
    return pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num, neg_num


'''Two Sample T'''
def T_stats_test(X,y, ridge = 1e-4):

    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num_array, neg_num_array = GetPosNegClassStats(X,y)
    T_stat_list = []
    for i in range(len(pos_mean_array)):
        pos_mean, neg_mean, pos_var, neg_var, pos_num, neg_num = \
            pos_mean_array[i], neg_mean_array[i], pos_var_array[i], neg_var_array[i], pos_num_array[i], neg_num_array[i]
        
        T_test = np.absolute(pos_mean - neg_mean) / np.sqrt(pos_var/pos_num + neg_var/neg_num + ridge)
        T_stat_list.append(T_test)
    
    T_stat_all = np.mean(T_stat_list)
    return T_stat_all

'''Absolute SNR '''
def Abs_SNR(X,y,ridge=1e-4):
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num_array, neg_num_array = GetPosNegClassStats(X,y)
    SNR_list = []
    for i in range(len(pos_mean_array)):
        pos_mean, neg_mean, pos_var, neg_var, pos_num, neg_num = \
         pos_mean_array[i], neg_mean_array[i], pos_var_array[i], neg_var_array[i], pos_num_array[i], neg_num_array[i]

        SNR = np.absolute(pos_mean - neg_mean) / (np.sqrt(pos_var) + np.sqrt(neg_var) + ridge)
        SNR_list.append(SNR)
    SNR_all = np.mean(SNR_list)
    return SNR_all

'''Symmetric Divergence'''
def SymmetricDivergence(X,y,ridge=0.05):
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num_array, neg_num_array = GetPosNegClassStats(X,y)
    SD_list = []
    for i in range(len(pos_mean_array)):
        pos_mean, neg_mean, pos_var, neg_var, pos_num, neg_num = \
         pos_mean_array[i], neg_mean_array[i], pos_var_array[i], neg_var_array[i], pos_num_array[i], neg_num_array[i]
        
        first_term = pos_var/(neg_var+ridge) + neg_var/(pos_var+ridge)
        second_term = ((pos_mean - neg_mean)**2) / (pos_var + neg_var + ridge)
        SD = 0.5 * first_term + 0.5 * second_term - 1
        
        SD_list.append(SD)
    SD_all = np.mean(SD_list)
    return SD_all


'''Fisher Discriminant Ratio'''
def FDR(X,y,ridge=1e-4):
    pos_mean_array, neg_mean_array, pos_var_array, neg_var_array, pos_num_array, neg_num_array = GetPosNegClassStats(X,y)
    FDR_list = []
    for i in range(len(pos_mean_array)):
        pos_mean, neg_mean, pos_var, neg_var, pos_num, neg_num = \
         pos_mean_array[i], neg_mean_array[i], pos_var_array[i], neg_var_array[i], pos_num_array[i], neg_num_array[i]

        FDR = np.square(pos_mean - neg_mean) / (pos_var + neg_var + ridge)
        FDR_list.append(FDR)
    FDR_all = np.mean(FDR_list)
    return FDR_all
