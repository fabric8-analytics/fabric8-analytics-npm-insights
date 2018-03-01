#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the logic for the autoencoder to generate
the package content encodings.

Copyright © 2018 Avishkar Gupta

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
from tensorflow.estimator import DNNClassifier


class CollabVariationalAE():
    """This class contains the collaborative variational
    autoencoder model definition."""

    def __init__(self, weights, bias):
        """Initialize the model with weights."""
        self._weights = weights
        self._bias = bias
        self._error = None
        self._optimize = None
        self._prediction = None

    @property
    def prediction(self):
        return self._prediction

    @property
    def optimize(self):
        return self._optimize

    @property
    def error(self):
        return self._error

    def generation_network(self):
        with tf.variable_scope('generation'):
            pass
