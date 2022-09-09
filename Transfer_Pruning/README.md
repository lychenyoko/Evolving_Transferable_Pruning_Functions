# Transfer Pruning with ResNet-56 on CIFAR-100

In this folder, we use the evolved functions to prune ResNet-56 on CIFAR-100, which is a totally different task than what the functions are evolved on. 

## Training Full-Size Original Model from Scratch

Training a full-size ResNet-56 on CIFAR-100 can be conducted by:

```
python3 model_training.py
```

This experiment is fully controlled by the hyper-parameters specified in the file [ResNet_hyperparams_training.py](./ResNet_hyperparams_training.py). 

