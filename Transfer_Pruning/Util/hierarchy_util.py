'''
This file provides util for conducting the class hierarchy learning
'''

import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def Get_Useful_Operation():
    '''
    Usage:
        Get the operation of a TensorFlow graph that is of interest
    '''
    
    node = [n for n in tf.get_default_graph().as_graph_def().node]
    op_tensor = [n for n in node if (n.input!=None and n.input!=[])]
    operation_list = [n for n in op_tensor if n.op!='Identity']
    
    return operation_list

def Get_ResNet_Last_Hidden_Layer_Tensor():
    '''
    Usage:
        Return the TensorFlow defined tensor in the last layer
    '''
    
    tensor_name = 'fc/Mean'
    last_layer_tenosr = tf.get_default_graph().get_tensor_by_name(name = tensor_name + ':0')
    
    return last_layer_tenosr
    
    
def Get_Class_Centroid_Dict(X,y):
    '''
    Usage:
        Return a dictionary with 
            key -> the index of the class
            val -> the centroid of the class data
    
    Args:
        X: (np.array) data matrix where each row is an instance
        y: (np.array) labels of each instance
    '''
    
    y_unique = np.unique(y)
    X_centroid_dict = {}
    for label in y_unique:
        key = label
        val = np.mean(X[np.where(y == label)[0]], axis = 0)
        X_centroid_dict[key] = val
    return X_centroid_dict


def Get_KMeans_Coarse_Fine_Dict(class_centroid_dict, num_coarse_classes):
    '''
    Usage:
        Using KMeans to find coarse/fine label mapping and return a dict:
            key -> coarse label
            val -> list of corresponding fine label
    
    Args:
        class_centroid_dict: (dict) with fine label as key and centroid vector as value
        num_coarse_classes: (int) # of coarse classes to group    
    '''
    
    # number of iteration for KMeans to run
    max_iter = 3000 
    # To change the dict to list for KMeans Clustering
    centroid_list = [value.reshape(-1) for value in list(class_centroid_dict.values())] 
    mKMeans = KMeans(n_clusters=num_coarse_classes, max_iter=max_iter).fit(centroid_list)
    
    Kmeans_coarse_fine_dict = defaultdict(list)
    # the fine label is just the index of the list(mKMeans.labels_)
    for fine, coarse in enumerate(list(mKMeans.labels_)):
        Kmeans_coarse_fine_dict[coarse].append(fine)
    
    return Kmeans_coarse_fine_dict


def Build_Acc_Confusion_Matrix(true_label, pred_label):
    '''
    Usage:
        Build a confusion matrix with: 
            M[true,pred] = occurence of true,label pair
        true, pred are in in the label space
    
    Return:
        The confusion matrix
    '''
    
    num_true_label = len(np.unique(true_label))
    num_pred_label = len(np.unique(pred_label))
    confusion_matrix = np.zeros((num_true_label, num_pred_label)) # initialized to 0
    for true, pred in zip(true_label, pred_label):
        confusion_matrix[true, pred] = confusion_matrix[true, pred] + 1
    
    return confusion_matrix


def See_Hierarchy_Grouping(coarse_fine_dict, fine_label_name):
    '''
    Usage:
        Visualize the hierarchy grouping in class names
    
    Args:
        coarse_fine_dict: (dict) with
                            key -> coarse label index
                            val -> iterable (list, set) of fine label index
        fine_label_name: (list) where i-th element is the name of the i-th class
    
    '''
    for coarse_ind in range(len(coarse_fine_dict.keys())):
        key = list(coarse_fine_dict.keys())[coarse_ind]
        fine_name_list = [fine_label_name[fine_ind] for fine_ind in coarse_fine_dict[key]]
        print('Group ' + str(coarse_ind) + ': ' + str(fine_name_list))

        
def Get_Permutation(coarse_fine_dict):
    '''
    Usage:
        Get the permutation for the confusion matrix by stacking the group of fine labels
        in the same coarse label
    
    Args:
        coarse_fine_dict: (dict) with key -> coarse_label, val -> fine_label
    '''
    
    permutation = []
    for key in sorted(coarse_fine_dict.keys()):
        val = list(coarse_fine_dict[key])
        permutation += val
    return permutation


def Permute_Similarity_Matrix(matrix, permutation):
    '''
    Usage:
        Permute the similarity matrix based on a permutation order,
        the permutation order is usually achieved with seriation algorithm
        the permutation is performed on both row and column axis
    
    Args:
        matrix: (np.array) square 2D matrix with each pixel showing similarity between classes
        permutation: (np.array or list) 1D sequence showing the permutation
    '''
    
    permuted_matrix = np.zeros(matrix.shape)
    for row in range(permuted_matrix.shape[0]):
        new_row = row
        ori_row = permutation[row] # permute in row axis
        permuted_matrix[new_row,:] = (matrix[ori_row,:])[permutation] # permute in column axis
    
    return permuted_matrix


def Log_Confusion_Matrix(confusion_matrix):
    '''
    Usage:
        Calculate the log value of the confusion matrix entry
        The entry value are non-negative integers 
        special instruction:
            0: still maps to 0
            1: maps to np.log(1.1) approximately equal to 0.1
    '''
    log_matrix = np.log(confusion_matrix)
    for i in range(log_matrix.shape[0]):
        for j in range(log_matrix.shape[1]):
            if log_matrix[i,j] == -np.inf: # 0 case
                log_matrix[i,j] = 0
            elif log_matrix[i,j] == 0: # 1 case
                log_matrix[i,j] = 0.1
    
    return log_matrix


def Transform_Confusion_to_Distance(confusion_matrix):
    '''
    Usage:
        Transform confusion matrix F into a distance matrix D,
        where 0 means classes are identical and higher values means less simlarity:
            D = 0.5 * [(I - F) + (I - F.T)] and D[i,i] = 0
    
    Args:
        confusion_matrix: (np.array) of 2D class confusion matrix with non-negative integer entry 
    '''
    
    transformed_confusion = 0.5 * (1 - confusion_matrix/100.0 + 1 - confusion_matrix.T/100)
    for i in range(transformed_confusion.shape[0]):
        transformed_confusion[i,i] = 0
    
    return transformed_confusion


def Get_Coarse_Fine_Dict_with_Sklearn_Cluster(cluster):
    '''
    Usage:
        Return a coarse to fine mapping dictionary with a sklearn cluster
    
    Args:
        cluster: (sklearn.cluster) where the cluster.labels indicate the fine class grouping
    
    '''
    coarse_fine_dict = defaultdict(list)
    # the fine label is just the index of the list(cluster.labels_)
    for fine, coarse in enumerate(list(cluster.labels_)):
        coarse_fine_dict[coarse].append(fine)
    return coarse_fine_dict


def Generate_Coarse_Label(coarse_fine_dict, dataset_fine_labels):
    '''
    Usage:
        Generate a list of coarse label based on the coarse_fine_dict and the fine_label_list
    
    Args:
        coarse_fine_dict: (dict) of key->coarse label and val->(list) of corresponding fine labels
        dataset_fine_labels: (list or np.array) 1D sequence of fine labels for the dataset
    '''
    
    # first build a reverse dictionary
    fine_coarse_dict = dict()
    for coarse_label,fine_label_list in coarse_fine_dict.items():
        for fine_label in fine_label_list:
            fine_coarse_dict[fine_label] = coarse_label

    dataset_coarse_labels = []
    for label in dataset_fine_labels:
        dataset_coarse_labels.append(fine_coarse_dict[int(label)])
    
    return dataset_coarse_labels



def Partition_Dataset_in_Class(X, y):
    '''
    Usage:
        To partition a dataset matrix into different classes in the ascending order of class index
        
    Args:
        X: (np.array) of 2D dataset matrix in dimension [N,D]
        y: (np.array) of label array in dimension [N] or [N,1]
    
    Note:
        We assume unique_y contains all the label in the dataset such that: 
            (1) unique_y = [0, num_class-1] 
            (2) partition_list[i] = data in class i
    
    '''
    
    unique_y = np.unique(y)
    partition_list = []
    for ylabel in unique_y:
        class_samples_index = np.where(y == ylabel)[0]
        class_samples = X[class_samples_index]
        partition_list.append(class_samples)
    return partition_list


def Build_DF_Matrix(class_partition, DF, m_ridge):
    '''
    Usage:
        Return a matrix DF_Matrix where DF_Matrix[i,j] = DF(class_i, class_j)
    
    Args:
        class_partition: (list) of class partition
        DF:      (func) a discriminant function as in the Util.Discriminant_Function
        m_ridge: (float) the ridge parameter for discriminant function
    '''
    
    class_num = len(class_partition)
    DF_Matrix = np.zeros((class_num, class_num))
    
    for i in range(class_num):
        for j in range(i + 1, class_num):
            class1 = class_partition[i]
            class2 = class_partition[j]
            two_class_dataset = np.concatenate((class1, class2), axis = 0)
            two_class_label = np.array([0] * class1.shape[0] + [1] * class2.shape[0])
            DF_Score = DF(two_class_dataset, two_class_label, m_ridge)
            DF_Matrix[i,j] = DF_Score
            DF_Matrix[j,i] = DF_Score    
    return DF_Matrix


def Sample_Variance(X):
    '''
    Usage:
        Calculate the sample variance of a dataset X
    
    Args:
        X: (np.array) 2D data array with dimension [N_samples, N_features]
    '''
    
    mX = np.mean(X, axis=0) # data centroid
    sample_diff = np.sum(np.square(X - mX), axis = 1)
    sample_var = np.mean(sample_diff)
    return sample_var


def Overall_Sample_Variance(X, y):
    '''
    Usage:
        Use sample variance as the quantitative measurement of cluster quality
        
    Args:
        X: (np.array) 2D data with dimension [N_samples, N_features]
        y: (np.array) of 1D data with dimension [N_samples] or [N_samples,1]
    '''
    
    y = np.array(y).reshape(-1)
    assert X.shape[0] == len(y)
    
    class_var_list = []
    uni_y = np.unique(y)
    for label in uni_y:
        class_ind = np.where(y == label)[0]
        class_data = X[class_ind,:]
        class_var = Sample_Variance(class_data)
        class_var_list.append(class_var)
    
    return np.mean(class_var_list)

def Average_Class_Centroid_Distance(X, y):
    '''
    Usage:
        Use class centroid distance as the quantitative measurement of cluster quality
        
    Args:
        X: (np.array) 2D data with dimension [N_samples, N_features]
        y: (np.array) of 1D data with dimension [N_samples] or [N_samples,1]
    '''

    y = np.array(y).reshape(-1)
    assert X.shape[0] == len(y)
    
    class_centroid_list = []
    uni_y = np.unique(y)
    for label in uni_y:
        class_ind = np.where(y == label)[0]
        class_centroid = np.mean(X[class_ind,:], axis=0)
        class_centroid_list.append(class_centroid)
    
    avg_distance = Get_Distance(class_centroid_list)
    return avg_distance

def Get_Distance(centroid_list):
    '''
    Usage:
        Get the centroid distance of all pairs
    
    Args:
        centroid_list: (list) of cluster centroid
    '''
    
    num_centroid = len(centroid_list)
    centroid_dist_list = []
    for i in range(num_centroid):
        for j in range(i+1, num_centroid):
            dist = np.linalg.norm(centroid_list[i] - centroid_list[j])
            centroid_dist_list.append(dist)
    
    avg_dist = np.mean(centroid_dist_list)
    return avg_dist