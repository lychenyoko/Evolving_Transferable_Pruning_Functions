# Evolving Transferable Neural Pruning Functions [[Paper]](https://dl.acm.org/doi/10.1145/3512290.3528694) [[ArXiv]](https://arxiv.org/abs/2110.10876)

## Paper in GECCO2022
```BibTex
@inproceedings{liu2022evolving,
  title     = {Evolving transferable neural pruning functions},
  author    = {Liu, Yuchen and Kung, Sun-Yuan and Wentzlaff, David},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  pages     = {385--394},
  year      = {2022}
}
```

This repository contains the codes for *Evolving Transferable Neural Pruning Functions*, which is published in GECCO.

## Overview

<a><img src='Assets/General_Framework.svg' width=1100></a>


We propose to learn transferable channel pruning functions via an evolution strategy, genetic programming. The evolved functions show generalizability when applied to datasets and networks different from that in evolution. 

## Usage

### Working Environment

We have tested our codes with the following setups:

```
python==3.6.5
tensorflow==1.14.0
numpy==1.16.0
```


### Evolution

We evolve the pool of functions on two pruning tasks: LeNet-5 on MNIST and VGG-16 on CIFAR-10. 
The execution codes are under the folder of [Evolution](./Evolution/). 

### Transfer Pruning 


The evolved functions can be used to conduct transfer pruning on datasets and networks that are different from the evolution. 
For instance, you can prune ResNet on CIFAR-100 with the evolved function (Equation 3 in the paper) under [Transfer_Pruning](./Transfer_Pruning/). 


