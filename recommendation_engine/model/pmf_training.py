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
import tensorflow as tf
import numpy as np
from recommendation_engine.config.params_training import training_params as training_params


class PMFTraining:
    """Training definitions the Probabilistic Matrix factorization model."""

    def __init__(self):
        """Create a new PMF training instance."""
        # TODO
        pass

    def __call__(self, *args, **kwargs):
        """Train the model."""
        min_iter = 1
        a_minus_b = training_params.a - training_params.b

    def save_model_to_s3(self):
        """Save the model in matlab format to load later for scoring."""
        # TODO
        pass
