#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameters used for training the model.

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

training_params = tf.contrib.training.HParams(
    num_latent=50,
    weight_decay=1e-05,
    learning_rate=0.001,
    batch_size=128,
    a=1,
    b=0.01,
    lambda_u=0.1,
    lambda_v=10,
    lambda_r=1,
    num_epochs=10,
    max_iter=10
)
