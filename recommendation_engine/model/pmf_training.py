#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains the code used to train the PMF part of CVAE.

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
import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

from recommendation_engine.config.params_training import training_params as training_params
import recommendation_engine.config.path_constants as path_constants


class PMFTraining:
    """Training definitions the Probabilistic Matrix factorization model."""

    def __init__(self, num_users, num_items, m_weights):
        """Create a new PMF training instance."""
        with tf.variable_scope('pmf_training', reuse=tf.AUTO_REUSE):
            self.m_items = tf.get_variable('m_V', initializer=tf.random_normal_initializer,
                                           shape=[num_users, training_params.num_latent])
            self.m_users = tf.get_variable('m_U', initializer=tf.random_normal_initializer,
                                           shape=[num_items, training_params.num_latent])
            self.m_weights = m_weights

    def __call__(self, *args, **kwargs):
        """Train the model."""
        convergence = tf.Variable(1.0, 'convergence', dtype=tf.float32)
        item_user_map = kwargs['item_to_user_matrix']
        user_item_map = kwargs['user_to_item_matrix']

        with tf.variable_scope('pmf_training', reuse=tf.AUTO_REUSE):
            a_minus_b = tf.subtract(training_params.a, training_params.b)
            likelihood = tf.Variable(-tf.exp(20.0), name='likelihood', dtype=tf.float32)
            likelihood_old = tf.Variable(0, name='likelihood_old', dtype=tf.float32)
            # Loop over the training till convergence
            for iteration in range(0, training_params.max_iter):
                likelihood = tf.assign(likelihood, 0)
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
                    item_norm = users_items_weighted + tf.matmul(
                        tf.gather(self.m_items, this_user_items),
                        tf.gather(self.m_items, this_user_items), transpose_a=True) * a_minus_b
                    tf.assign(self.m_users[user_id:], tf.linalg.solve(
                        item_norm,
                        training_params.a * tf.reduce_sum(tf.gather(self.m_items, this_user_items),
                                                          axis=0)))

                    likelihood = tf.assign(
                        likelihood,
                        likelihood + (-0.5) * training_params.lambda_u * tf.reduce_sum(
                            self.m_users[user_id, :] * self.m_users[user_id, :], axis=1))

                # Update the item vectors
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
                        self.m_items[item_id, :] = tf.linalg.solve(
                            item_norm,
                            training_params.lambda_v * self.m_weights[item_id, :])

                        # now calculate the likelihood
                        epsilon = self.m_items[item_id, :] - self.m_weights[item_id, :]
                        likelihood += -0.5 * training_params.lambda_v * tf.reduce_sum(
                            tf.square(epsilon))
                    else:
                        item_norm = items_users_weighted + tf.matmul(
                            tf.gather(self.m_users, this_item_users),
                            tf.gather(self.m_users, this_item_users), transpose_a=True) * a_minus_b

                        item_norm_pre_weighing = item_norm
                        item_norm += tf.eye(training_params.num_latent) * training_params.lambda_v

                        tf.assign(self.m_items[item_id:], tf.linalg.solve(
                            item_norm,
                            training_params.a * tf.reduce_sum(
                                tf.gather(self.m_users, this_item_users),
                                axis=0) + training_params.lambda_v * self.m_weights[item_id, :]))

                        likelihood += (-0.5) * len(user_ids) * training_params.a + \
                            training_params.a * tf.reduce_sum(
                                tf.matmul(tf.gather(self.m_users, user_ids),
                                          tf.reshape(self.m_items[item_id, :],
                                                     [training_params.num_latent, 1])), axis=0)

                        likelihood += -0.5 * tf.matmul(
                            tf.matmul(self.m_items[item_id, :], item_norm_pre_weighing),
                            tf.reshape(self.m_items[item_id, :], [training_params.num_latent, 1]))

                        epsilon = self.m_items[item_id, :] - self.m_weights[item_id, :]
                        likelihood += (-0.5) * training_params.lambda_v * tf.reduce_sum(
                            tf.square(epsilon))

                iteration += 1
                convergence = tf.assign(convergence,
                                        tf.abs((likelihood - likelihood_old) / likelihood_old))
                if convergence < 1e-6 and iteration > training_params.min_iter:
                    break

    def save_model(self, data_store):
        """Save the model in matlab format to load later for scoring."""
        with tf.Session() as session:
            local_file_path = os.path.join(path_constants.LOCAL_MODEL_DIR,
                                           path_constants.PMF_MODEL_PATH)
            savemat(local_file_path,
                    {"m_U": session.run(self.m_users),
                     "m_V": session.run(self.m_items),
                     "m_theta": session.run(self.m_weights)})
            data_store.upload_file(local_file_path)
