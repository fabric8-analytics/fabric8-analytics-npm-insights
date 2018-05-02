#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to handle pre-train and training of the model.

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

from recommendation_engine.model.collaborative_variational_autoencoder import \
    CollaborativeVariationalAutoEncoder
from recommendation_engine.model.pmf_training import PMFTraining


class TrainingJob:
    """Define an instance of a training job for the model."""

    @classmethod
    def train(cls):
        """Fire a training job."""
        CollaborativeVariationalAutoEncoder.train(cls.train_input_fn)
        pmf_training = PMFTraining()
        pmf_training()
        # TODO

    @staticmethod
    def train_input_fn():
        """Pass the input to the estimator for training."""
        # TODO
        pass
