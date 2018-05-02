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


def add_hidden_layer_summary(val_to_write):
    """Write summary of hidden layer to file for tensorboard."""
    tf.summary.scalar('zero_values_in_hidden_layer', tf.nn.zero_fraction(val_to_write))
    tf.summary.histogram('activation', val_to_write)


def inference_network(inputs, hidden_units, is_training, scope=None):
    """Layer definition for the encoder layer of CVAE."""
    net = inputs
    with tf.variable_scope('inference_network'):
        for hidden_dim in hidden_units:
            net = tf.contrib.layers.fully_connected(
                net,
                num_outputs=hidden_dim,
                scope='fc-{}-output-neurons'.format(hidden_dim))
    return net


def generation_network(inputs, hidden_units, is_training, scope=None):
    """Define the decoder network of CVAE."""
    net = inputs  # inputs here is the latent representation.
    for i, hidden_dim in enumerate(hidden_units, 1):
        with tf.variable_scope('layer_{}'.format(i), values=(net,)):
            net = tf.contrib.layers.fully_connected(
                net,
                num_outputs=hidden_dim,
                initializer=tf.contrib.layers.xavier_initializer)
            add_hidden_layer_summary(net)
        generated = tf.nn.softmax_cross_entropy_with_logits(net)
    generated = tf.identity(generated, name='reconstructed')
    return generated


def _autoencoder_arg_scope(activation_fn, is_training):
    """Create an argument scope for the network based on its parameters."""
    weights_regularizer = tf.contrib.layers.l2_regularizer()

    with slim.arg_scope([tf.contrib.layers.fully_connected],
                        weights_initializer=tf.initializers.variance_scaling(),
                        weights_regularizer=weights_regularizer,
                        activation_fn=activation_fn) as arg_sc:
        return arg_sc


def cvae_autoencoder_net(inputs, hidden_units, activation=tf.nn.sigmoid, weight_decay=None,
                         mode=tf.estimator.ModeKeys.TRAIN, scope=None):
    """Create the CVAE network layers.

    :inputs: tf.Tensor holding the input data.
    :hidden_units: list[int] - containing number of nodes in each hidden layer.
    :weight_decay: float - Amount of regularization to use on the weights(excludes biases).
    mode : tf.estimator.ModeKeys - The mode of the model.
    scope : str - Name to use in Tensor board.

    :returns: tf.Tensor - Output of the generation network's reconstruction layer.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope(scope, 'AutoEnc', [inputs]):
        with slim.arg_scope(
                _autoencoder_arg_scope(activation, weight_decay, mode)):
            net = inference_network(inputs, hidden_units, is_training)
            n_features = inputs.shape[1].values
            decoder_units = hidden_units[:-1][::-1] + [n_features]
            net = generation_network(net, decoder_units, is_training)

    return net
