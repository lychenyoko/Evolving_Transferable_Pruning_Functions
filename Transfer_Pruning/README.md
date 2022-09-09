# Transfer Pruning with ResNet-56 on CIFAR-100

In this folder, we use the evolved functions to prune ResNet-56 on CIFAR-100, which is a totally different task than what the functions are evolved on. 

## Training Full-Size Original Model from Scratch

Training a full-size ResNet-56 on CIFAR-100 can be conducted by:

```
python3 model_training.py
```

This experiment is fully controlled by the hyper-parameters specified in the file [ResNet_hyperparams_training.py](./ResNet_hyperparams_training.py). 

## Evolved-Function Transfer Pruning

With the full-size model file (referred as `trained_ResNet_ckpt.pkl` here) saved from the above training from scratch experiment, we can perform transfer pruning by:

```
python3 model_pruning.py \
	--model=trained_ResNet_ckpt.pkl
```

The hyper-parameter setting of this experiment is fully specified in the file of [ResNet_hyperparams_pruning.py](./ResNet_hyperparams_pruning.py). 
By default, we use the evolved function in `Equation 3` of the main paper as the channel pruning criteria. 

