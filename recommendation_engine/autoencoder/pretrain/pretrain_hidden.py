#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pretrain and return the autoencoder hidden layers."""

import logging
import os
import daiquiri
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python import keras
import recommendation_engine.config.params_training as train_params
from recommendation_engine.config.path_constants import TEMPORARY_MODEL_PATH
from recommendation_engine.utils.fileutils import check_path
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
        # If that changes in some version, explicitly set a kernel and bias initializer.
        layer_network = keras.layers.Dense(self.layer_dim, activation=train_params.activation,
                                           name='pretrain_hidden_{}'.format(self.layer_dim))
        encoder = layer_network(enc_inp_noisy)
        encoder = keras.models.Model(enc_inp, encoder)
        layer_network = layer_network(enc_inp_noisy)
        layer_network = keras.layers.Dense(input_dimension, activation=train_params.activation,
                                           name="de")(layer_network)
        layer_model = keras.models.Model(enc_inp, layer_network)
        layer_model.compile(optimizer=tf.optimizers.Adam(train_params.learning_rate_pretrain),
                            loss=train_params.loss_hidden)
        path = os.path.join(TEMPORARY_MODEL_PATH, 'hidden_{}_pretrain/'.format(self.layer_dim))
        saver = keras.callbacks.ModelCheckpoint(
            check_path(path),
            save_weights_only=True,
            verbose=1)
        tensorboard_config = TensorBoard(log_dir=check_path(path))

        layer_model.fit(data, data, batch_size=batch_size, epochs=epochs,
                        callbacks=[saver, tensorboard_config])
        # append the weight of the layer being pre-trained.
        self.layer_weights = layer_model.get_layer(
                "pretrain_hidden_{}".format(self.layer_dim)).get_weights()
        self.de_weights = layer_model.get_layer("de").get_weights()
        return encoder.predict(data, batch_size=batch_size)

    def get_layer_weights(self):
        """Get the weights of this layer."""
        return self.layer_weights, self.de_weights
