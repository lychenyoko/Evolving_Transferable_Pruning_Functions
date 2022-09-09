# Define the Primitive and the Operation for Getting Primitives Arg

import numpy as np
from deap import gp
from .Self_Defined_Op import *
import itertools
chain_list = lambda c: list(itertools.chain.from_iterable(c))

# --------------------------------- The Pset Used For A Bunch of Discriminant Functions ---------------------------------

# Operands
num_inp = 9
Single_Tree_Pset = gp.PrimitiveSet('Single_Tree_Set',arity=num_inp)
Single_Tree_Pset.renameArguments(ARG0 = 'P', ARG1 = 'Q', ARG2 = 'N', ARG3 = 'X', ARG4 = 'mX', ARG5 = 'mP', ARG6 = 'mQ', ARG7 = 'WI', ARG8 = 'W')
P,Q,N,X,mX,mP,mQ,WI,W = list(Single_Tree_Pset.terminals.values())[0] 
''' 
P: Positive Matrix 
Q: Negative Matirx 
N: Number of Samples 
X: Whole Data Matrix 
mX: Data Vector Mean 
mP: Vector Mean of Positive Matrix 
mQ: Vector Mean of Negative Matrix
WI: Channel's incoming filter of shape [Nin, H, W]
W: Whole layer's filter of shape [Nin, Nout, H, W]
''' 

# Operators
Single_Tree_Pset.addPrimitive(m_trace,arity=1)
Single_Tree_Pset.addPrimitive(m_matmul,arity=2,name='mmul')
Single_Tree_Pset.addPrimitive(m_inv,arity=1)
Single_Tree_Pset.addPrimitive(np.add,arity=2)
Single_Tree_Pset.addPrimitive(m_sub,arity=2,name='sub')
Single_Tree_Pset.addPrimitive(m_mul,arity=2,name='mul')
Single_Tree_Pset.addPrimitive(np.outer,arity=2,name='out_dot')
Single_Tree_Pset.addPrimitive(np.transpose,arity=1,name='tran')

Single_Tree_Pset.addPrimitive(np.mean,arity=1,name='g_mean')
Single_Tree_Pset.addPrimitive(m_mean,arity=1,name='mean')
Single_Tree_Pset.addPrimitive(np.sum,arity=1,name='sum')
Single_Tree_Pset.addPrimitive(np.std,arity=1,name='std')
Single_Tree_Pset.addPrimitive(np.var,arity=1,name='var')
Single_Tree_Pset.addPrimitive(np.divide,arity=2,name='div')
Single_Tree_Pset.addPrimitive(np.absolute,arity=1,name='abs')
Single_Tree_Pset.addPrimitive(np.square,arity=1,name='pow2')
Single_Tree_Pset.addPrimitive(len,arity=1,name='num')
Single_Tree_Pset.addPrimitive(np.sqrt,arity=1,name='sqrt')

Single_Tree_Pset.addPrimitive(m_rho1,arity=1,name='rho1')
Single_Tree_Pset.addPrimitive(m_rho5,arity=1,name='rho5')
Single_Tree_Pset.addPrimitive(Get_Kernel_GM,arity=1,name='geo_median')

trace,matmul,inv,add,sub,mul,out_dot,tran,\
glb_mean,axi_mean,glb_sum,std,var,div,ab,pow2,num,sqrt,rho1,rho5,geo_median = list(Single_Tree_Pset.primitives.values())[0]


# --------------------------------- Defining Individual for Initial Population ---------------------------------

# AbsSNR
AbsSNR = [div, ab, sub, glb_mean, P, glb_mean, Q, rho1, add, std, P, std, Q]

FDR = [div, pow2, sub, glb_mean, P, glb_mean, Q, rho1, add, var, P, var, Q]

def GetSymDiv():
    first_term = [add, div, var, P, rho5, var, Q, div, var, Q, rho5, var, P]
    second_term = [div, pow2, sub, glb_mean, P, glb_mean, Q, rho5, add, var, P, var, Q]
    SymDiv = chain_list([[add],first_term,second_term])
    return SymDiv

def GetTstats():
    numer = [ab, sub, glb_mean, P, glb_mean, Q]
    deno  = [sqrt, rho1, add, div, var, P, num, P, div, var, Q, num, Q]
    Tstats = chain_list([[div],numer,deno])
    return Tstats

WL1 = [glb_sum, ab, WI]
WL2 = [sqrt, glb_sum, pow2, WI]

GeoMedian = [sqrt, glb_sum, pow2, sub, WI, geo_median, W]


# --------------------------------- Self Define Class to Represent an Individual for Evolution ---------------------------------
def arg_for_DF(X,y):
    '''
    Usage:
        To obtain the operand for feature map-based channel scoring functions

    Args:
        X: (np.array) of feature map tensor with dimension [N, H, W] or [N] 
        y: (np.array) of labels with dimension [N] or [N, 1]
    '''
    X = X.reshape(X.shape[0],-1)
    mx = np.mean(X,axis=0)    
    class_ , class_count = np.unique(y,return_counts = True)
    pos_X_list = []
    neg_X_list = []
    for label in class_:
        pos_X_list.append(X[np.where(y==label)[0]])
        neg_X_list.append(X[np.where(y!=label)[0]])
    return [pos_X_list,neg_X_list,list(class_count),X,mx]


class DFunc_Individual:
    def __init__(self, pset, expr_tree):
        self.pset = pset
        self.expr_tree = gp.PrimitiveTree(expr_tree)
        self.fitness = None

def Metric_Score(func, arg):
    '''
    Usage:
        Return the score of a GP function on the argument
    
    Args:
        func: (gp.func) a compiled gp function
        arg: (list) including feature maps and filters associated to the channel
    '''
    Plist,Qlist,Nlist,X,mX,WI,W = arg
    
    score_list = []
    for P,Q,N in zip(Plist,Qlist,Nlist):
        P = P.reshape(P.shape[0],-1)
        mP = np.mean(P, axis = 0) # positive data mean
        Q = Q.reshape(Q.shape[0],-1)
        mQ = np.mean(Q, axis = 0)
        
        score = func(P,Q,N,X,mX,mP,mQ,WI,W)
        score_list.append(score)

    return np.sum(np.array(score_list))



## DI
#M_list = []
#for pos_class,num_class in zip([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9],[N0,N1,N2,N3,N4,N5,N6,N7,N8,N9]):
#    M = [mul,out_dot,sub,mX,axi_mean,pos_class,sub,mX,axi_mean,pos_class,num_class]
#    M_list.append(M)
#    
#SB = []
#for i in range(len(M_list)-1):
#    SB.append([add])
#    SB.append(M_list[i])
#SB.append(M_list[-1])
#SB = chain_list(SB)
#SS = [matmul,tran,sub,X,mX,sub,X,mX]
#
#DI = chain_list([[trace],[matmul],[inv],SS,SB])    