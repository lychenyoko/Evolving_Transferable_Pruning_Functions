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
    '''
    Usage:
        Get the Discriminant Information Score

    Input:
        X: (np.array) of dimension [N] or [N,D]
        y: (np.array) of dimension [N] or [N,1]
    '''

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


def GetPosNegClassStats(X,y):
    '''
    Usage:  
        Base Function to Obtain Statistics that Will be Used for FDR, Abs_SNR, SymDiv, TwoT
    
    Args:        
        X: (np.array) of dimension 2, [N, D]
        y: (np.array) od dimension either [N, 1] or [N]
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

# ========================================= Implementing the Maximum Mean Discrepancy =========================================

def Get_RBF_Kernel_Marix(X, sigma = 1):
    '''
    Usage:
        Obtain the kernel matrix based on the input data
    
    Args:
        X: (np.array) 2D array of data matrix with dimension [N, D]
        sigma: (float) the parameter for Gaussian Kernel
    '''
        
    num_sample = X.shape[0]
    Euclidean_Distance_Matrix = np.zeros((num_sample, num_sample))
    for i in range(num_sample):
        for j in range(i+1, num_sample):
            Euclidean_Distance_Matrix[i,j] = np.sum((X[i,...] - X[j, ...])**2)
            Euclidean_Distance_Matrix[j,i] = Euclidean_Distance_Matrix[i,j]
    
    Kernel_Matrix = np.exp( -Euclidean_Distance_Matrix / (2 * sigma**2) )
    return Kernel_Matrix


def Two_Class_MMD(Kernel_Matrix, Class1_Ind, Class2_Ind):
    '''
    Usage:
        Return the two class MMD score
    
    Args:
        Kernel_Matrix: (np.array) dataset's 2D kernel matrix with dimension [N,N]
        Class1_Ind: (np.array or list) 1D index list of the class1 indices
        Class2_Ind: (np.array or list) 1D index list of the class2 indices
    '''
    class1_term = np.mean(Kernel_Matrix[Class1_Ind, :][:, Class1_Ind])
    class2_term = np.mean(Kernel_Matrix[Class2_Ind, :][:, Class2_Ind])
    cross_term = np.mean(Kernel_Matrix[Class1_Ind, :][:, Class2_Ind])
    
    two_class_MMD = class1_term + class2_term - 2 * cross_term
    return two_class_MMD


def Multi_Class_MMD(X, y, sigma=1):
    '''
    Usage:
        Return the multi-class MMD score with a one-versus-rest implementation
        
    Args:
        X: (np.array) of 2D data matrix with dimension [N_sample, N_feature]
        y: (np.array) of label with dimension [N_sample] or [N_sample, 1]
        sigma: (float) sigma of the Gaussian Kernel for calculation
    
    '''    
    
    if len(X.shape) == 1: # Reshape 1D array
        X = X.reshape(-1, 1)
    
    KMatrix = Get_RBF_Kernel_Marix(X, sigma)
    
    MMD_score_list = []
    label_list = np.unique(y)
    for label in label_list:
        positive_class_ind = np.where(y == label)[0]
        negative_class_ind = np.where(y != label)[0]
        MMD_score = Two_Class_MMD(KMatrix, positive_class_ind, negative_class_ind)
        MMD_score_list.append(MMD_score)
    
    Final_MMD = np.mean(MMD_score_list)
    
    return Final_MMD

def Minibatch_MMD(X, y, sigma=1, batch_size=100):
    '''
    Usage:
        To estimate the data's MMD in a shuffled minibatch way
        
    Args:
        X:          (np.array) of 2D data matrix with dimension [N_sample, N_feature]
        y:          (np.array) of label with dimension [N_sample] or [N_sample, 1]
        sigma:      (float) sigma of the Gaussian Kernel for calculation
        batch_size: (int) the size of each estimation batch
    '''
    
    num_batch = len(y) // batch_size
    MMD_list = []
    for batch_id in range(num_batch):
        mb_x, mb_y = X[batch_id * batch_size: (batch_id + 1) * batch_size], y[batch_id * batch_size: (batch_id + 1) * batch_size]
        MMD_score = Multi_Class_MMD(mb_x, mb_y, sigma)
        MMD_list.append(MMD_score)
    
    return MMD_score
