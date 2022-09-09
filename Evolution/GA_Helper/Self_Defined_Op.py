# The Self Defined Function

import numpy as np

def m_trace(m_array):
    if np.isscalar(m_array):
        return m_array
    
    elif len(m_array.shape) == 1:
        return m_array[0]
        
    else:
        m_array = np.array(m_array)
        dim = np.min(m_array.shape)
        return np.trace(m_array[:dim,:dim])

# Implement a ridged inversion
def m_inv(m_array):
    if np.isscalar(m_array):
        return 1/m_array
    
    elif len(m_array.shape) == 1:
        return 1/m_array[0]
    
    else:
        m_array = np.array(m_array)
        dim = np.min(m_array.shape)
        return np.linalg.pinv(m_array[:dim,:dim] + 1e-4 * np.eye(dim))

def m_add(op1,op2):
    if np.isscalar(op1) or np.isscalar(op2):
        return np.add(op1,op2)
    
    else:
        op1,op2 = np.array(op1),np.array(op2)
        dim = np.min((op1.shape[-1],op2.shape[-1]))
        return np.add(op1[...,:dim],op2[...,:dim])

def m_sub(op1,op2):
    if np.isscalar(op1) or np.isscalar(op2):
        return np.subtract(op1,op2)
    
    else:
        op1,op2 = np.array(op1),np.array(op2)
        dim = np.min((op1.shape[-1],op2.shape[-1]))
        return np.subtract(op1[...,:dim],op2[...,:dim])

def m_matmul(op1,op2):
    if np.isscalar(op1) or np.isscalar(op2):
        return np.multiply(op1,op2)
    else:
        op1,op2 = np.array(op1),np.array(op2)
        dim = np.min((op1.shape[-1],op2.shape[0]))
        return np.matmul(op1[...,:dim],op2[:dim])

def m_mul(op1,op2):
    if np.isscalar(op1) or np.isscalar(op2):
        return np.multiply(op1,op2)
    else:
        op1,op2 = np.array(op1),np.array(op2)
        dim = np.min((op1.shape[-1],op2.shape[0]))
        return np.multiply(op1[...,:dim],op2[:dim])

def m_mean(X):
    if X.shape[1] > 1:
        return np.mean(X,axis=0)    
    return np.mean(X)

def m_rho1(X):
    return np.add(X,1e-4)

def m_rho5(X):
    return np.add(X,5e-2)


def Get_Kernel_GM(mKer):
    '''
    Args:
        mKer: (np.array) a 4D Kernel [w, h, inp, out]
    
    Return:
        GM: (np.array) a 3D Kernel [w, h, inp] which is the geometric median of 
    '''
    
    Layer_GM_Score = []
    for out_ind in range(mKer.shape[-1]):
        channel_gm_score = Get_Channel_GM_Score(mKer = mKer, index = out_ind)
        Layer_GM_Score.append(channel_gm_score)
    
    GM_index = np.argmin(Layer_GM_Score)
    GM = mKer[..., GM_index]
    return GM


def Get_Channel_GM_Score(mKer, index):
    '''
    Args:
        mKer: (np.array) a 4D Kernel [w, h, inp, out]
        index: (int) the index of the last dimension of mKer for GM Score calculation
    '''
    
    mask = [True] * mKer.shape[-1]
    mask[index] = False
    
    sel_ker = mKer[..., index]
    other_ker = mKer[..., mask]
    
    norm_score = 0
    for i in range(other_ker.shape[-1]):
        diff = sel_ker - other_ker[..., i]
        tmp = np.linalg.norm(diff.reshape(-1))
        norm_score = norm_score + tmp

    return norm_score