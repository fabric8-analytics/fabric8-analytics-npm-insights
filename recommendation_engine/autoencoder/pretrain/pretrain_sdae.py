#!/usr/bin/env python
# encoding: utf-8
"""Pretrain the VAE using a SDAE."""

import logging
import daiquiri
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Dense, GaussianNoise, Input, Lambda
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.models import Model
from recommendation_engine.utils.fileutils import check_path
import recommendation_engine.config.params_training as params_training
from recommendation_engine.config.path_constants import TEMPORARY_SDAE_PATH

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class PretrainNetwork:
    """Pretrain the autoencoder in stacked denoinsing autoencoder fashion."""

    def __init__(self, hidden_dim, pretrain):
        """Construct an object with following properties."""
        self.weights = []
        self.pretrain = pretrain
        self.hidden_dim = hidden_dim

    @staticmethod
    def sampling(args):
        """Define a sampling for our lambda layer.

        Taken from:
        https://github.com/keras-team/keras/master/examples/variational_autoencoder.py
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def train(self, data):
        """Pretrain the latent layers of the model."""
        # network parameters
        original_dim = data.shape[1]
        input_shape = (original_dim,)
        batch_size = params_training.batch_size
        latent_dim = params_training.num_latent
        epochs = params_training.num_epochs
        layer_num = 0

        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        inputs_noisy = GaussianNoise(stddev=0.1)(inputs)
        hidden = inputs_noisy
        for i, hidden_dim in enumerate(self.hidden_dim, 1):
            hidden_layer = Dense(hidden_dim, activation='sigmoid', name='hidden_e_{}'.format(i),
                                 weights=self.pretrain[layer_num])
            hidden = hidden_layer(hidden)
            layer_num += 1
            logger.debug("Hooked up hidden layer with %d neurons" % hidden_dim)
        z_mean = Dense(latent_dim, activation=None, name='z_mean',
                       weights=self.pretrain[layer_num])(hidden)
        layer_num += 1
        z_log_sigma = Dense(latent_dim, activation=None, name='z_log_sigma',
                            weights=self.pretrain[layer_num])(hidden)
        layer_num += 1
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_sigma])
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        hidden = latent_inputs
        for i, hidden_dim in enumerate(self.hidden_dim[::-1], 1):  # Reverse because decoder.
            hidden = Dense(hidden_dim, activation='sigmoid', name='hidden_d_{}'.format(i),
                           weights=self.pretrain[layer_num])(hidden)
            layer_num += 1
            logger.debug("Hooked up hidden layer with %d neurons" % hidden_dim)
        outputs = Dense(original_dim, activation='sigmoid')(hidden)
        decoder = Model(latent_inputs, outputs, name='decoder')

        # Build the DAE
        outputs = decoder(encoder(inputs)[2])
        sdae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = binary_crossentropy(inputs, outputs) * original_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        sdae.add_loss(vae_loss)
        sdae.compile(optimizer='adam')
        saver = ModelCheckpoint(
            check_path(TEMPORARY_SDAE_PATH),
            save_weights_only=True,
            verbose=1)
        tensorboard_config = TensorBoard(log_dir=check_path(TEMPORARY_SDAE_PATH))
        logger.info("Checkpoint has been saved for SDAE.")
        # train the autoencoder
        logger.warning("Pretraining started, Don't interrupt.")
        sdae.fit(data, epochs=epochs, batch_size=batch_size,
                 callbacks=[saver, tensorboard_config])
        logger.info("Model has been pretrained successfully.")
