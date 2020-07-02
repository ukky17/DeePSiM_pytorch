# Pytorch implementation of DeePSiM

This repository contains a Pytorch-based implementation of DeePSiM [[Dosovitskiy & Brox, 2016](https://arxiv.org/abs/1602.02644)]. The code here is desigined for inverting a VGG16-like network trained on CIFAR10 dataset, but you can modify slightely to work for other network architectures and datasets.

## Tested environment
* Python 3.6.6
* Pytorch 1.0.1

## Usage
1. run `ff_training.py` to train the feed-forward network on CIFAR10 (input --> target)
2. run `get_rpr.py` to obtain the internal representation at the specified layer (input --> hidden).
3. run `invert.py` to invert the internal representation back to the input space (hidden --> input).

## Example results
![exmample](example.png)

## Reference
Generating Images with Perceptual Similarity Metrics based on Deep Networks https://arxiv.org/abs/1602.02644.

Hyperparameter values are obtained with reference to the above paper and the [Tensorflow implementation](https://github.com/shijx12/DeepSim).
