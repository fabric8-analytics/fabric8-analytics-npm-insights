#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the Sampling layer that makes the sampling layer.

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
import recommendation_engine.config.params_training as config


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = tf.random_normal((config.batch_size, config.num_latent), 0, 1,
                               dtype=tf.float32)
    print(z_mean + tf.exp(z_log_sigma) * epsilon)
    return z_mean + tf.exp(z_log_sigma) * epsilon
