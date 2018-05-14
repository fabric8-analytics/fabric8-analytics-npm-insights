#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the CVAE model definition.

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
from recommendation_engine.model.layer_definitions import cvae_autoencoder_net
from recommendation_engine.config.params_training import training_params


def cvae_net_model_fn(features, labels, hidden_units, output_dim, activation, learning_rate, mode):
    """Model function for the CVAE estimator."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Define model's architecture
    logits = cvae_autoencoder_net(inputs=features,
                                  hidden_units=hidden_units,
                                  output_dim=output_dim,
                                  activation=activation,
                                  mode=mode,
                                  scope='VarAutoEnc')

    probs = tf.nn.sigmoid(logits)
    predictions = {"prediction": probs}

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Estimator spec for prediction (runtime).
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    tf.losses.sigmoid_cross_entropy(labels, logits)
    tf.losses.add_loss(tf.losses.get_regularization_loss())
    total_loss = tf.losses.get_total_loss(add_regularization_losses=is_training)

    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            optimizer="Adam",
            learning_rate=learning_rate,
            learning_rate_decay_fn=lambda lr, gs: tf.train.exponential_decay(lr, gs, 1000, 0.96,
                                                                             staircase=True),
            global_step=tf.train.get_global_step(),
            summaries=["learning_rate", "global_gradient_norm"])

        # Add histograms for trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Provide an estimator spec for `ModeKeys.TRAIN` mode.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op)


class CollaborativeVariationalAutoEncoder(tf.estimator.Estimator):
    """Estimator API wrapper for CVAE autoencoder model."""

    def __init__(self, hidden_units, output_dim, activation_fn=tf.nn.sigmoid,
                 learning_rate=training_params.learning_rate, model_dir=None, config=None):
        """Create a new CVAE estimator."""
        def _model_fn(features, labels, mode):
            return cvae_net_model_fn(
                features=features,
                labels=labels,
                hidden_units=hidden_units,
                output_dim=output_dim,
                activation=activation_fn,
                learning_rate=learning_rate,
                mode=mode)

        super().__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config)
