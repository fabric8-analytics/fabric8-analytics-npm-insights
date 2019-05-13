#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pretrain and return the autoencoder hidden layers."""

import logging

import daiquiri
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.callbacks import TensorBoard

import recommendation_engine.config.params_training as train_params
import recommendation_engine.config.path_constants_train as path_constants

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class PretrainHidden:
    """Pretrain the hidden layers and stores the layer weights."""

    def __init__(self, layer_dim):
        """Create a new pretraining instance."""
        self.layer_weights = None
        self.de_weights = None
        self.layer_dim = layer_dim

    def train(self, data, input_dimension, epochs=train_params.num_epochs,
              batch_size=train_params.batch_size):
        """Pretrain the hidden layers."""
        logger.debug("Training layer with dimension {}".format(self.layer_dim))
        logger.debug("Shape of data is: {}".format(data.shape))
        enc_inp = keras.layers.Input(shape=(input_dimension,))
        enc_inp_noisy = keras.layers.GaussianNoise(stddev=0.1)(enc_inp)
        # Default initializer is glorot uniform, and default bias initializer is 0,
        # If that changes in some version, explicity set a kernel and bias initializer.
        layer_network = keras.layers.Dense(self.layer_dim, activation=train_params.activation,
                                           name='pretrain_hidden_{}'.format(self.layer_dim))
        encoder = layer_network(enc_inp_noisy)
        encoder = keras.Model(enc_inp, encoder)
        layer_network = layer_network(enc_inp_noisy)

        layer_network = keras.layers.Dense(input_dimension, activation=train_params.activation,
                                           name="de")(
                layer_network)
        layer_model = keras.Model(enc_inp, layer_network)
        layer_model.compile(optimizer=tf.train.AdamOptimizer(train_params.learning_rate_pretrain),
                            loss='binary_crossentropy')
        saver = keras.callbacks.ModelCheckpoint(
                '/tmp/hidden_{}_pretrain/train'.format(self.layer_dim),
                save_weights_only=True,
                verbose=1)
        tensorboard_config = TensorBoard(log_dir=str(
                path_constants.TENSORBOARD_LOGDIR_LOCAL.joinpath(
                        'pretrain_hidden_{}'.format(self.layer_dim))))
        layer_model.fit(data, data, batch_size=batch_size, epochs=epochs,
                        callbacks=[tensorboard_config, saver])
        # append the weight of the layer being pre-trained.
        self.layer_weights = layer_model.get_layer(
                "pretrain_hidden_{}".format(self.layer_dim)).get_weights()
        self.de_weights = layer_model.get_layer("de").get_weights()

        return encoder.predict(data, batch_size=batch_size)

    def get_layer_weights(self):
        """Get the weights of this layer."""
        return self.layer_weights, self.de_weights
