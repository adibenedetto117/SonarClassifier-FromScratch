# SonarClassifier-FromScratch

## Introduction

`SonarClassifier-FromScratch` is a pure Python implementation of a neural network, aimed at classifying sonar signals. This project has been developed without the use of high-level machine learning libraries such as TensorFlow or PyTorch. Instead, it relies solely on NumPy to handle matrix operations, with every part of the neural network's logic—forward and backpropagation, activation functions, and loss computation—explicitly coded from the ground up.

## Motivation

The motivation behind creating a neural network from scratch is to gain an intricate understanding of the learning processes and mathematical operations that constitute machine learning models. This approach broke down the complex algorithms and provides insights into how models learn and make predictions, which is often not shown by machine learning frameworks.

## Technical Overview

This project's neural network architecture is a simple feedforward model with two hidden layers. The key operations are as follows:

- **Activation Functions**: Implemented the Rectified Linear Unit (ReLU) for hidden layers and the sigmoid function for the output layer, crucial for binary classification tasks.
- **Loss Function**: The log loss, also known as binary cross-entropy, measures the performance of the classification model.
- **Weight Initialization**: Applied the He initialization method, which is specifically designed for layers with ReLU activation, to help in the effective training of deep networks.
- **Backpropagation**: Coded the backpropagation algorithm from scratch to compute the gradient of the loss function with respect to each weight and bias in the network.
