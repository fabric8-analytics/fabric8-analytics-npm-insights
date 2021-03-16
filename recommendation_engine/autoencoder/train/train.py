#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the CVAE model."""

import os
import daiquiri
import logging
import numpy as np
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.models import Model
from recommendation_engine.pmf.train_pmf import PMFTraining
import recommendation_engine.config.params_training as params_training
from recommendation_engine.autoencoder.pretrain import pretrain
from recommendation_engine.config.path_constants import TEMPORARY_SDAE_PATH, TEMPORARY_CVAE_PATH, \
    TEMPORARY_USER_ITEM_FILEPATH, TEMPORARY_ITEM_USER_FILEPATH, TEMPORARY_PATH, \
    TEMPORARY_MODEL_PATH, TEMPORARY_DATASTORE, TEMPORARY_DATA_PATH
from recommendation_engine.utils.fileutils import check_path, load_rating
daiquiri.setup(level=logging.DEBUG)
from training.datastore.get_preprocess_data import GetPreprocessData
from training.datastore.s3_helper import S3Helper
from training.datastore.npm_metadata import NPMMetadata
logger = daiquiri.getLogger(__name__)


class TrainNetwork:
    """Train the autoencoder(CVAE)."""

    def __init__(self, hidden_dim=(200, 100), pretrain_weights=True,
                 aws_access_key_id=os.environ.get("AWS_S3_ACCESS_KEY_ID", ""),
                 aws_secret_access_key=os.environ.get("AWS_S3_SECRET_ACCESS_KEY",
                                                      ""),
                 aws_bucket_name=os.environ.get("AWS_S3_BUCKET_NAME", "cvae-insights"),
                 model_version=os.environ.get("MODEL_VERSION", ""),
                 num_train_per_user=os.environ.get("TRAIN_PER_USER", 5)
                 ):
        """Construct an object with following properties."""
        self.weights = []
        self.pretrain_weights = pretrain_weights
        self.hidden_dim = hidden_dim
        self.get_preprocess_data = GetPreprocessData(aws_access_key_id,
                                                     aws_secret_access_key,
                                                     aws_bucket_name,
                                                     model_version,
                                                     num_train_per_user
                                                     )

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

    @staticmethod
    def load_pretrain_weights(model):
        """Load the weights from the pre-training job."""
        model.load_weights(TEMPORARY_SDAE_PATH)
        logger.info(
            "Successfully loaded weights from: {}".format(TEMPORARY_SDAE_PATH))
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
        z_mean = Dense(params_training.num_latent, activation=None, name='z_mean')(hidden)
        z_log_sigma = Dense(params_training.num_latent, activation=None, name='z_log_sigma')(hidden)
        z = Lambda(self.sampling, output_shape=(params_training.num_latent,), name='z')(
                [z_mean, z_log_sigma])
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        self.encoder_z_mean = encoder.predict(data)[0]

        # build decoder model
        latent_inputs = Input(shape=(params_training.num_latent,), name='z_sampling')
        hidden = latent_inputs
        for i, hidden_dim in enumerate(self.hidden_dim[::-1], 1):  # Reverse because decoder.
            hidden = Dense(hidden_dim, activation='sigmoid', name='hidden_d_{}'.format(i))(hidden)
            logger.debug("Hooked up hidden layer with %d neurons" % hidden_dim)
        # if hidden == latent_inputs:
        #     logger.warning("No Hidden layers hooked up.")
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
                check_path(TEMPORARY_CVAE_PATH),
                save_weights_only=True,
                verbose=1
        )
        tensorboard_config = TensorBoard(log_dir=check_path(TEMPORARY_CVAE_PATH))
        # train the auto-encoder
        cvae_model.fit(data, epochs=params_training.num_epochs,
                       batch_size=params_training.batch_size,
                       callbacks=[saver, tensorboard_config])
        return self.encoder_z_mean


if __name__ == '__main__':
    p = TrainNetwork()
    logger.info("Preprocessing of data started.")
    p.get_preprocess_data.preprocess_data()
    x_train = np.load(os.path.join(TEMPORARY_DATA_PATH, 'content_matrix.npz'))
    x_train = x_train['matrix']
    input_dim = x_train.shape[1]
    logger.info("size of training file is: {}, {}".format(len(x_train), len(x_train[0])))
    user_to_item_matrix = load_rating(TEMPORARY_USER_ITEM_FILEPATH, TEMPORARY_DATASTORE)
    item_to_user_matrix = load_rating(TEMPORARY_ITEM_USER_FILEPATH, TEMPORARY_DATASTORE)
    logger.info("Shape of User and Item matrices: {} , {}".format(np.shape(user_to_item_matrix),
                                                                  np.shape(item_to_user_matrix)))
    pretrain.fit(x_train)
    encoder_weights = p.train(x_train)
    logger.info("Shape of encoder weights are: {}, {}, {}".format(tf.shape(encoder_weights),
                                                                  len(encoder_weights),
                                                                  len(encoder_weights[0])))
    pmf_obj = PMFTraining(len(user_to_item_matrix), len(item_to_user_matrix), encoder_weights)
    logger.debug("PMF model has been initialised")
    pmf_obj(user_to_item_matrix=user_to_item_matrix,
            item_to_user_matrix=item_to_user_matrix)
    logger.debug("PMF model has been trained.")
    pmf_obj.save_model()
    p.get_preprocess_data.obj_.save_on_s3(TEMPORARY_DATA_PATH)
    p.get_preprocess_data.obj_.save_on_s3(TEMPORARY_PATH)
    p.get_preprocess_data.obj_.save_on_s3(TEMPORARY_MODEL_PATH)
