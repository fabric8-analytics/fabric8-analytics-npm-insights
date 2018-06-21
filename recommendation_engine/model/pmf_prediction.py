#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the code that deals with the PMF piece for scoring.

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
import numpy as np
import scipy

from recommendation_engine.config.params_scoring import ScoringParams


class PMFScoring:
    """This class defines the PMF scoring logic.

    This is decoupled from the training logic to run inside openshift without using tensorflow.
    """

    def __init__(self, model_dict, items):
        """Create an instance of PMF scoring."""
        self.params = ScoringParams()
        self.m_V = model_dict["m_V"]
        self.m_U = model_dict["m_U"]
        self.items = items

    def predict_transform(self, user_item_vector, num_latent):
        """Create this users' m_U vector.

        Create the factor space mapping for this user, this
        can then be multiplied by the latent item space mapping
        to get the recommendations for this user.

        :user_vector: A list containing the items in the users' stack.
        :num_latent: The number of latent factors to use.
        :returns: A numpy array containing the user vector calculated
                  based on the latent item vectors.
        """
        # VV^T for v_j that has at least one user liked
        x = self.params.a * np.sum(self.m_V[user_item_vector, :], axis=0)
        return scipy.linalg.solve(self.params.a, x).reshape(1, num_latent)
