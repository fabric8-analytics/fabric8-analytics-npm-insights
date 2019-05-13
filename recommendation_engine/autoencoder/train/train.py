#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the CVAE model."""
import logging

import daiquiri
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.models import Model

import recommendation_engine.config.params_training as params_training
from recommendation_engine.config import path_constants_train as path_constants

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class TrainNetwork:
    """Train the autoencoder(CVAE)."""

    def __init__(self, hidden_dim=(200, 100), pretrain_weights=True):
        """Constructor to define data stores etc."""
        self.weights = []
        self.pretrain_weights = pretrain_weights
        self.hidden_dim = hidden_dim

    @staticmethod
    def sampling(args):
        """Define a sampling for our lambda layer.

        Taken from:
        https://github.com/keras-team/keras/master/examples/variational_autoencoder.py"""
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    @staticmethod
    def load_pretrain_weights(model):
        """Load the weights from the pre-training job."""
        #  TODO: Load from S3
        model.load_weights(params_training.PRETRAIN_WEIGHTS_PATH)
        logger.info(
            "Successfully loaded weights from: {}".format(params_training.PRETRAIN_WEIGHTS_PATH))
        return model

    def train(self, data):
        """Pretrain the latent layers of the model."""
        # network parameters
        original_dim = data.shape[1]
        input_shape = (original_dim,)

        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        hidden = inputs
        for i, hidden_dim in enumerate(self.hidden_dim, 1):
            hidden = Dense(hidden_dim, activation='sigmoid', name='hidden_e_{}'.format(i))(hidden)
            logger.debug("Hooked up hidden layer with %d neurons" % hidden_dim)
        if hidden == inputs:
            logger.warning("No Hidden layers hooked up.")
        z_mean = Dense(params_training.num_latent, activation=None, name='z_mean')(hidden)
        z_log_sigma = Dense(params_training.num_latent, activation=None, name='z_log_sigma')(hidden)
        z = Lambda(self.sampling, output_shape=(params_training.num_latent,), name='z')(
                [z_mean, z_log_sigma])
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(params_training.num_latent,), name='z_sampling')
        hidden = latent_inputs
        for i, hidden_dim in enumerate(self.hidden_dim[::-1], 1):  # Reverse because decoder.
            hidden = Dense(hidden_dim, activation='sigmoid', name='hidden_d_{}'.format(i))(hidden)
            logger.debug("Hooked up hidden layer with %d neurons" % hidden_dim)
        if hidden == latent_inputs:
            logger.warning("No Hidden layers hooked up.")
        outputs = Dense(original_dim, activation='sigmoid')(hidden)
        decoder = Model(latent_inputs, outputs, name='decoder')

        # Build the CVAE auto-encoder
        outputs = decoder(encoder(inputs)[2])
        cvae_model = Model(inputs, outputs, name='cvae')
        # Load the pre-trained weights.
        self.load_pretrain_weights(cvae_model)
        reconstruction_loss = binary_crossentropy(inputs, outputs) * original_dim
        kl_loss = 1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)

        cvae_model.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
        cvae_model.compile(optimizer='adam')
        cvae_model.summary()

        # First load the weights from the pre-training
        if self.pretrain_weights:
            cvae_model = self.load_pretrain_weights(cvae_model)

        saver = ModelCheckpoint(
                '/tmp/train/train',
                save_weights_only=True,
                verbose=1)
        tensorboard_config = TensorBoard(log_dir=str(
                path_constants.TENSORBOARD_LOGDIR_LOCAL.joinpath(
                        'cvae_train')))

        # train the auto-encoder
        cvae_model.fit(data, epochs=params_training.num_epochs,
                       batch_size=params_training.batch_size,
                       callbacks=[saver, tensorboard_config])

        self.cvae_model = cvae_model

    def compute_latent_embeddings(self, ip_samples):
        """Compute the latent embeddings for all the packages."""
        self.cvae_model.predict(ip_samples)


if __name__ == '__main__':
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    p = TrainNetwork()
    p.train(x_train)
