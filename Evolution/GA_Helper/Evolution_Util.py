import numpy as np
import subprocess as cmd

def find_ind_from_name(file, model_name):
    '''
    Usage:
        To help sort the file list by the individual #
    '''
    assert model_name in ['LeNet5','VGG16']
    start_ind = len(model_name) + 5
    end_ind = file.find('_', start_ind)
    ind_num = int(file[start_ind:end_ind])
    return ind_num

def LeNet5_find_ind_from_name(file):
    return find_ind_from_name(file, 'LeNet5')


def VGG16_find_ind_from_name(file):
    return find_ind_from_name(file, 'VGG16')

     
# Used for getting fitness for each individual
def get_fitness(file):
    fit_dict = {}
    
    acc_start = file.find('acc_') + 4
    acc_end = file.find('_FLOP')
    fit_dict['acc'] = float(file[acc_start:acc_end])

    FLOP_start = file.find('FLOP_') + 5
    FLOP_end = file.find('_Param')
    fit_dict['FLOP'] = round(100 - float(file[FLOP_start:FLOP_end]),2)
    
    Param_start = file.find('Param_') + 6
    Param_end = file.find('.npy')
    fit_dict['Param'] = round(100 - float(file[Param_start:Param_end]),2)
    
    return fit_dict 


# Combining two tasks performance for a single fitness for the child
def weighted_geometric_mean(a, b, alpha):
    '''
    Args:
        a,b : two operands
        alpha: a parameter between 0 and 1
    '''
    mean = np.power(a, alpha) * np.power(b, (1 - alpha))
    return mean

def weighted_arithmetic_mean(a, b, alpha):
    '''
    Args:
        a,b : two operands
        alpha: a parameter between 0 and 1
    '''
    mean = a * alpha +  b * (1 - alpha)
    return mean

def joint_fitness(VGG16_acc, LeNet5_acc, alpha, method):
    '''
    Usage:
        A unified method for getting the joint fitness
    '''
    if method == 'weighted_geometric_mean':
        return weighted_geometric_mean(VGG16_acc, LeNet5_acc, alpha)
    
    if method == 'weighted_arithmetic_mean':
        return weighted_arithmetic_mean(VGG16_acc, LeNet5_acc, alpha)


# Evaluate LeNet5
def Evaluate_Ind_LeNet5(slurm_out, ind_file, g, i, Res_dir, exp_time, slurm_dir):
    m_file_name = './LeNet5_Evaluation_' + str(g) + '_' + str(i) + '_'+ exp_time + '.slurm'
    m_file = open(m_file_name,'w')

    script = '''#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --nodelist=adroit-h11g1
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=yl16@princeton.edu
#SBATCH --output=%s
#SBATCH --mem=12000
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.5.0

python3 Co-Evolution_LeNet5_VGG16/LeNet5_Eval_Parallel.py --indv=%s --gen=%s --i=%s --dir=%s
'''%(slurm_out,ind_file,g,i,Res_dir)

    m_file.write(script)
    m_file.close()

    command = ['sbatch', m_file_name]
    cmd.run(command)
    print('Run command: '+ str(command))

    # Move the .slurm file into the slurm directory
    mv_slurm_cmd = ['mv', m_file_name, slurm_dir]
    cmd.run(mv_slurm_cmd)


# Evaluate VGG16
def Evaluate_Ind_VGG16(slurm_out, ind_file, g, i, Res_dir, exp_time, slurm_dir):
    m_file_name = './VGG16_Evaluation_' + str(g) + '_' + str(i) + '_'+ exp_time + '.slurm'
    m_file = open(m_file_name,'w')

    script = '''#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --nodelist=adroit-h11g1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=yl16@princeton.edu
#SBATCH --output=%s
#SBATCH --mem=80000
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.5.0

python3 Co-Evolution_LeNet5_VGG16/VGG16_Eval_Parallel.py --indv=%s --gen=%s --i=%s --dir=%s
'''%(slurm_out,ind_file,g,i,Res_dir)

    m_file.write(script)
    m_file.close()

    command = ['sbatch', m_file_name]
    cmd.run(command)
    print('Run command: '+ str(command))

    # Move the .slurm file into the slurm directory
    mv_slurm_cmd = ['mv', m_file_name, slurm_dir]
    cmd.run(mv_slurm_cmd)

