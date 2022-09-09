import numpy as np

# LeNet-5 Calculators
def GetReduction(net):
    c1,c2,f1,f2 = [int(i) for i in net]
    c1_pr = c1 / 20
    c2_pr = c1 * c2 / (20 * 50)
    f1_pr = f1 * f2 / (800 * 500)
    f2_pr = f2 / 500
    return c1_pr, c2_pr, f1_pr, f2_pr

ori_FLOP = 4586
def LeNet5_FLOPCal(net):
    conv1_FLOP = 576
    conv2_FLOP = 3200
    fc1_FLOP = 800
    fc2_FLOP = 10
    c1_pr, c2_pr, f1_pr, f2_pr = GetReduction(net)
    pruned_FLOP = c1_pr * conv1_FLOP + c2_pr * conv2_FLOP + f1_pr * fc1_FLOP + f2_pr * fc2_FLOP
    return pruned_FLOP/ori_FLOP 

ori_param = 430.5
def LeNet5_ParamCal(net):
    conv1_pm = 0.5
    conv2_pm = 25
    fc1_pm = 400
    fc2_pm = 5
    c1_pr, c2_pr, f1_pr, f2_pr = GetReduction(net)
    pruned_pm = c1_pr * conv1_pm + c2_pr * conv2_pm + f1_pr * fc1_pm + f2_pr * fc2_pm
    return pruned_pm/ori_param  

