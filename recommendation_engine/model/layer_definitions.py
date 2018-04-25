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


def inference_network(inputs, hidden_units, dropout, is_training, scope=None):
    """Layer definition for the encoder layer of CVAE."""
    # TODO
    net = inputs
    return net


def generation_network(inputs, hidden_units, dropout, is_training, scope=None):
    """Define the decoder network of CVAE."""
    # TODO
    net = inputs
    return net


def _autoencoder_arg_scope(activation_fn, dropout, weight_decay, is_training):
    if weight_decay is None or weight_decay <= 0:
        weights_regularizer = None
    else:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with slim.arg_scope([tf.contrib.layers.fully_connected],
                        weights_initializer=tf.initializers.variance_scaling(),
                        weights_regularizer=weights_regularizer,
                        activation_fn=activation_fn):
        with slim.arg_scope([slim.dropout],
                            keep_prob=dropout,
                            is_training=is_training) as arg_sc:
            return arg_sc


def cvae_autoencoder_net(inputs, hidden_units, activation=tf.nn.sigmoid, dropout=None,
                         weight_decay=None,
                         mode=tf.estimator.ModeKeys.TRAIN, scope=None):
    """Create the CVAE network layers.

    :inputs: tf.Tensor holding the input data.
    :hidden_units: list[int] - containing number of nodes in each hidden layer.
    :dropout : float - Dropout value if using.
    :weight_decay: float - Amount of regularization to use on the weights(excludes biases).
    mode : tf.estimator.ModeKeys - The mode of the model.
    scope : str - Name to use in Tensor board.

    :returns: tf.Tensor - Output of the generation network's reconstruction layer.
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope(scope, 'AutoEnc', [inputs]):
        with slim.arg_scope(
                _autoencoder_arg_scope(activation, dropout, weight_decay, mode)):
            net = inference_network(inputs, hidden_units, dropout, is_training)
            n_features = inputs.shape[1].value
            decoder_units = hidden_units[:-1][::-1] + [n_features]
            net = generation_network(net, decoder_units, dropout, is_training)

    return net
