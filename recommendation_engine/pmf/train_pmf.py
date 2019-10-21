#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the probabilistic matrix factorization model."""

import os
import logging
import daiquiri
import tensorflow as tf
import numpy as np
from scipy.io import savemat
import recommendation_engine.config.params_training as training_params
import recommendation_engine.config.path_constants as path_constants
from recommendation_engine.utils.fileutils import check_path

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)


class PMFTraining:
    """Training definitions the Probabilistic Matrix factorization model."""

    def __init__(self, num_users, num_items, mat_weights):
        """Create a new PMF training instance."""
        with tf.compat.v1.variable_scope('pmf_training', reuse=tf.compat.v1.AUTO_REUSE):
            self.m_items = tf.compat.v1.get_variable('m_V',
                                                     initializer=tf.random_normal_initializer,
                                                     shape=[num_items, training_params.num_latent],
                                                     dtype=tf.float32)

            self.m_users = tf.compat.v1.get_variable('m_U',
                                                     initializer=tf.random_normal_initializer,
                                                     shape=[num_users, training_params.num_latent],
                                                     dtype=tf.float32)

            self.m_weights = tf.compat.v1.get_variable('m_Theta',
                                                       initializer=mat_weights,
                                                       dtype=tf.float32)

    def __call__(self, **kwargs):
        """Train the model."""
        item_user_map = kwargs['item_to_user_matrix']
        user_item_map = kwargs['user_to_item_matrix']
        with tf.compat.v1.variable_scope('temp_training', reuse=tf.compat.v1.AUTO_REUSE):
            a_minus_b = tf.subtract(tf.constant(training_params.a, dtype=tf.float32),
                                    tf.constant(training_params.b, dtype=tf.float32))
            likelihood = tf.compat.v1.get_variable('likelihood', initializer=-tf.exp(20.0),
                                                   dtype=tf.float32)
            likelihood_old = tf.compat.v1.get_variable('likelihood', initializer=0.0,
                                                       dtype=tf.float32)
            # Loop over the training till convergence
            convergence = tf.compat.v1.get_variable('convergence', initializer=1e-6,
                                                    dtype=tf.float32)
            for iteration in range(0, training_params.max_iter):
                tf.compat.v1.assign(likelihood_old, likelihood)
                likelihood = tf.convert_to_tensor(0.0)

                # Update the user vectors.
                item_ids = np.array([np.array(idx) for idx, row in enumerate(item_user_map)
                                     if len(row) > 0])
                rated_items = tf.gather(self.m_items, item_ids)
                user_items_sq = tf.matmul(rated_items, rated_items, transpose_a=True)
                users_items_weighted = user_items_sq * training_params.b + tf.eye(
                    training_params.num_latent) * training_params.lambda_u

                for user_id, this_user_items in enumerate(user_item_map):
                    if len(this_user_items) == 0:
                        continue
                    temp = tf.matmul(
                        tf.gather(self.m_items, this_user_items, axis=0),
                        tf.gather(self.m_items, this_user_items, axis=0),
                        transpose_a=True) * a_minus_b
                    item_norm = users_items_weighted + temp
                    temp_ = training_params.a * tf.reduce_sum(
                        tf.gather(self.m_items, this_user_items), axis=0)
                    temp_ = temp_[:, None]
                    user_val_temp = tf.linalg.solve(
                        item_norm,
                        temp_)
                    user_val_temp = tf.reshape(user_val_temp, [50])
                    tf.compat.v1.assign(self.m_users[user_id, :],
                                        user_val_temp)
                    # It will not update the value of users matrix

                    likelihood = likelihood + (-0.5) * training_params.lambda_u * tf.reduce_sum(
                                    self.m_users[user_id, :] * self.m_users[user_id, :])

                user_ids = np.array([np.array(idx) for idx, row in enumerate(user_item_map)
                                     if len(row) > 0])
                all_items_user = tf.gather(self.m_users, user_ids)
                items_users_weighted = tf.matmul(all_items_user, all_items_user,
                                                 transpose_a=True) * training_params.b

                for item_id, this_item_users in enumerate(item_user_map):
                    if len(this_item_users) == 0:
                        # Never been rated
                        item_norm = items_users_weighted + tf.eye(
                            training_params.num_latent) * training_params.lambda_v
                        weights_temp = training_params.lambda_v * self.m_weights[item_id, :]
                        weights_temp = weights_temp[:, None]

                        tf.compat.v1.assign(self.m_items[item_id, :], tf.reshape(tf.linalg.solve(
                            item_norm, weights_temp), [50]))
                        epsilon = self.m_items[item_id, :] - self.m_weights[item_id, :]
                        likelihood += -0.5 * training_params.lambda_v * tf.reduce_sum(
                            tf.square(epsilon))
                    else:
                        item_norm = items_users_weighted + tf.matmul(
                            tf.gather(self.m_users, this_item_users),
                            tf.gather(self.m_users, this_item_users),
                            transpose_a=True) * a_minus_b

                        item_norm_pre_weighing = item_norm
                        item_norm += tf.eye(training_params.num_latent) * training_params.lambda_v
                        item_temp = training_params.a * tf.reduce_sum(
                            tf.gather(self.m_users, this_item_users),
                            axis=0) + training_params.lambda_v * self.m_weights[item_id, :]
                        item_val_temp = tf.linalg.solve(
                            item_norm, item_temp[:, None])
                        item_val_temp = tf.reshape(item_val_temp, [50])
                        tf.compat.v1.assign(self.m_items[item_id, :], item_val_temp)
                        likelihood += (-0.5) * len(user_ids) * training_params.a + (
                                      training_params.a * tf.reduce_sum(
                                                    tf.matmul(tf.gather(self.m_users, user_ids),
                                                              tf.reshape(self.m_items[item_id, :],
                                                              [training_params.num_latent, 1])),
                                                    axis=0
                                                                        )
                                                                                    )
                        likelihood = tf.reshape(likelihood, [])
                        m_items_temp = tf.reshape(self.m_items[item_id, :],
                                                  [training_params.num_latent, 1])
                        temp_matmul = tf.matmul(m_items_temp, item_norm_pre_weighing,
                                                transpose_a=True)
                        likelihood += -0.5 * tf.matmul(
                            temp_matmul,
                            tf.reshape(self.m_items[item_id, :],
                                       [training_params.num_latent, 1]))
                        likelihood = tf.reshape(likelihood, [])
                        epsilon = self.m_items[item_id, :] - self.m_weights[item_id, :]
                        help_value = (-0.5) * training_params.lambda_v * tf.reduce_sum(
                            tf.square(epsilon))
                        likelihood = likelihood + help_value

                logger.info("Likelihood and Likelihood old are"
                            " respectively: {}, {}".format(likelihood,
                                                           likelihood_old))
                convergence = tf.abs((likelihood - likelihood_old) / likelihood_old)
                convergence_val = convergence
                logger.info("Convergence is: {}".format(convergence_val))
                if convergence_val < 1e-6 and iteration > training_params.min_iter_pmf:
                    logger.info("PMF model is converging.")
                    break

    def save_model(self, data_store=None):
        """Save the model in matlab format to load later for scoring."""
        local_file_path = os.path.join(path_constants.TEMPORARY_PMF_PATH)
        local_file_path = check_path(local_file_path)
        savemat(local_file_path,
                {"m_U": self.m_users,
                 "m_V": self.m_items,
                 "m_theta": self.m_weights
                 }
                )
        if data_store:
            data_store.upload_file(local_file_path)
