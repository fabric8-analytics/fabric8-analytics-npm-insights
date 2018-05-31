#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the definitions for the network layers.

Copyright © 2018 Red Hat Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
from recommendation_engine.config.params_training import training_params


def add_layer_summary(val_to_write):
    """Write summary of hidden layer to file for tensorboard."""
    tf.summary.scalar('zero_values_in_hidden_layer', tf.nn.zero_fraction(val_to_write))
    tf.summary.histogram('activation', val_to_write)


def inference_network(inputs, hidden_units, n_outputs):
    """Layer definition for the encoder layer of CVAE."""
    print("Hidden units: {}".format(hidden_units))
    net = inputs
    with tf.variable_scope('inference_network', reuse=tf.AUTO_REUSE):
        for layer_idx, hidden_dim in enumerate(hidden_units):
            net = layers.fully_connected(
                net,
                num_outputs=hidden_dim,
                weights_regularizer=layers.l2_regularizer(training_params.weight_decay),
                scope='inf_layer_{}'.format(layer_idx))
            add_layer_summary(net)
        z_mean = layers.fully_connected(net, num_outputs=n_outputs, activation_fn=None,
                                        scope='inf_layer_{}_mean'.format(len(hidden_units)))
        z_log_sigma = layers.fully_connected(
            net, num_outputs=n_outputs, activation_fn=None,
            scope='inf_layer_{}_log_sigma'.format(len(hidden_units)))

    # margin of error
    epsilon = tf.random_normal(
        (training_params.batch_size, training_params.num_latent), 0, 1, seed=0)
    latent_representation = z_mean + tf.sqrt(tf.maximum(tf.exp(z_log_sigma), 1e-10)) * epsilon
    return latent_representation


def generation_network(inputs, decoder_units, n_x):
    """Define the decoder network of CVAE."""
    net = inputs  # inputs here is the latent representation.
    with tf.variable_scope("generation_network", reuse=tf.AUTO_REUSE):
        assert (len(decoder_units) >= 2)
        # First layer does not have a regularizer
        net = slim.fully_connected(
            net,
            decoder_units[0],
            scope="gen_layer_0",
        )
        for idx, decoder_unit in enumerate([decoder_units[1], n_x], 1):
            net = slim.fully_connected(
                net,
                decoder_unit,
                scope="gen_layer_{}".format(idx),
                weights_regularizer=layers.l2_regularizer(training_params.weight_decay)
            )
    # TODO: Generalize this bit to figure out how to auto add more layers
    # Assign the transpose of weights to the respective layers
    print(tf.trainable_variables())
    tf.assign(tf.get_variable("generation_network/gen_layer_1/weights"),
              tf.transpose(tf.get_variable("inference_network/inf_layer_1/weights")))
    tf.assign(tf.get_variable("generation_network/gen_layer_1/biases"),
              tf.get_variable("inference_network/inf_layer_0/biases"))
    tf.assign(tf.get_variable("generation_network/gen_layer_2/weights"),
              tf.transpose(tf.get_variable("inference_network/inf_layer_0/weights")))
    return net  # x_recon


def _autoencoder_arg_scope(activation_fn):
    """Create an argument scope for the network based on its parameters."""

    with slim.arg_scope([layers.fully_connected],
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=tf.initializers.constant(0.0),
                        activation_fn=activation_fn) as arg_sc:
        return arg_sc


def cvae_autoencoder_net(inputs, hidden_units, activation=tf.nn.sigmoid,
                         output_dim=training_params.num_latent,
                         mode=tf.estimator.ModeKeys.TRAIN, scope=None):
    """Create the CVAE network layers.

    :inputs: tf.Tensor holding the input data.
    :hidden_units: list[int] - containing number of nodes in each hidden layer.
    :mode: tf.estimator.ModeKeys - The mode of the model.
    :scope: str - Name to use in Tensor board.

    :returns: tf.Tensor - Output of the generation network's reconstruction layer.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if not is_training:
        # TODO: Go to evaluation/scoring based on this parameters' value
        pass
    inputs = tf.Print(inputs, [inputs], first_n=1)
    with tf.variable_scope(scope, 'AutoEnc', [inputs], reuse=tf.AUTO_REUSE):
        with slim.arg_scope(_autoencoder_arg_scope(activation)):
            latent_representation = inference_network(inputs, hidden_units,
                                                      n_outputs=output_dim)
            n_features = inputs.shape[1].value
            net = generation_network(latent_representation, hidden_units[::-1], n_features)
    return net
