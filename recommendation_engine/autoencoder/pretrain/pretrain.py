#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import daiquiri
import numpy as np

import recommendation_engine.autoencoder.pretrain.pretrain_hidden as pretrain_hidden
import recommendation_engine.autoencoder.pretrain.pretrain_latent as pretrain_latent
import recommendation_engine.autoencoder.pretrain.pretrain_sdae as pretrain_vae
import recommendation_engine.config.params_training as params

daiquiri.setup(level=logging.DEBUG)
logr = daiquiri.getLogger("pretrain")


def fit(data):
    # First pretrain the hidden layer
    # Output of previous layer for the input layer is the data itself.
    prev_layer_op = data
    weights = []
    de_weights = []
    for hidden_dim in params.hidden_dims:
        logr.debug("Pretraining hidden layer with dim: %d" % hidden_dim)
        temp_hidden_layer = pretrain_hidden.PretrainHidden(hidden_dim)
        prev_layer_op = temp_hidden_layer.train(prev_layer_op, np.shape(prev_layer_op)[1])
        logr.debug("Previous layer output has shape: {}".format(prev_layer_op.shape))
        enc_weight, de_weight = temp_hidden_layer.get_layer_weights()
        weights.append(enc_weight)
        de_weights.append(de_weight)
    # Now pretrain the latent layer.
    latent = pretrain_latent.PretrainLatent()
    latent.train(prev_layer_op)
    latent_weights, de_latent = latent.get_weights()
    weights += latent_weights
    de_weights += de_latent
    vae_pre_ob = pretrain_vae.PretrainNetwork(params.hidden_dims, weights + de_weights[::-1])
    vae_pre_ob.train(data)


if __name__ == '__main__':
    from tensorflow.python.keras.datasets import mnist

    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape([60000, 784]) / 255.
    fit(train_x)
