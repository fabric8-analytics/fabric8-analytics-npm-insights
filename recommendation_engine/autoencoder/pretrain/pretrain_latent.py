# encoding: utf-8
"""Pretrain the latent layers."""

import logging
import daiquiri
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.models import Model
import recommendation_engine.config.params_training as train_params
from recommendation_engine.config.path_constants import TEMPORARY_LATENT_PATH
from recommendation_engine.utils.fileutils import check_path

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class PretrainLatent:
    """Pretrain the layers interacting with the  latent space."""

    def __init__(self):
        """Construct an object with following properties."""
        self.weights = []
        self.de_weights = []

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
        batch_size = train_params.batch_size
        latent_dim = train_params.num_latent
        epochs = train_params.num_epochs
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        inputs_noisy = inputs
        z_mean = Dense(latent_dim, activation=None, name='z_mean')
        z_mean = z_mean(inputs_noisy)
        z_log_sigma = Dense(latent_dim, activation=None, name='z_log_sigma')
        z_log_sigma = z_log_sigma(inputs_noisy)
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_sigma])
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        outputs = Dense(original_dim, activation='sigmoid', name="decoder_l")(latent_inputs)
        decoder = Model(latent_inputs, outputs, name='decoder')
        # Build the DAE
        outputs = decoder(encoder(inputs)[2])
        latent_model = Model(inputs, outputs, name='vae_mlp')
        reconstruction_loss = binary_crossentropy(inputs, outputs) * original_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        latent_model.add_loss(vae_loss)
        latent_model.compile(optimizer='adam')
        saver = ModelCheckpoint(
            check_path(TEMPORARY_LATENT_PATH),
            save_weights_only=True,
            verbose=1
        )
        tensorboard_config = TensorBoard(log_dir=check_path(TEMPORARY_LATENT_PATH))
        logger.info("Model checkpoints has ben saved.")
        # train the autoencoder
        latent_model.fit(data, epochs=epochs, batch_size=batch_size,
                         callbacks=[saver, tensorboard_config])
        # Collect the weights for z_log_sigma and z_mean, the layers being pretrained.
        self.weights.append(latent_model.get_layer("encoder").get_layer("z_mean").get_weights())
        self.weights.append(
                latent_model.get_layer("encoder").get_layer("z_log_sigma").get_weights())
        self.de_weights.append(
                latent_model.get_layer("decoder").get_layer("decoder_l").get_weights())
        logger.info("Weights has been updated successfully.")

    def get_weights(self):
        """Get the weights of the latent dimension."""
        return self.weights, self.de_weights
