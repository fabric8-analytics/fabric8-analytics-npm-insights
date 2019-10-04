#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hyperparameters used for training the model."""

num_latent = 50
weight_decay = 1e-05
learning_rate_pretrain = 0.01
learning_rate_train = 0.001
batch_size = 256
a = 1
b = 0.01
lambda_u = 0.1
lambda_v = 10
lambda_r = 1
num_epochs = 1
max_iter = 5
hidden_dims = [200, 100]
loss_hidden = 'binary_crossentropy'
activation = 'sigmoid'
min_iter_pmf = 1
