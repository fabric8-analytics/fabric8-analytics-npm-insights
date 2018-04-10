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


class PMFScoring(object):
    """This class defines the PMF scoring logic.

    This is decoupled from the training logic to run inside openshift without using tensorflow.
    """

    def __init__(self, model_dict, items):
        """Create an instance of PMF scoring."""
        self.params = ScoringParams()
        self.m_V = model_dict["m_V"]
        self.m_U = model_dict["m_U"]
        self.items = items

    def predict_transform(self, user_vector):
        """
        Transform the user vector to a NXD vector.

        :user_vector: A list containing the items in the users' stack.
        :items: The item to user matrix.
        :params: An instance of the parameters object.

        :returns: A numpy array containing the user vector calculated
                  based on the latent item vectors.
        """
        a_minus_b = self.params.a - self.params.b

        # VV^T for v_j that has at least one user liked
        ids = np.array([len(x) for x in self.items]) > 0
        v = self.m_V[ids]
        VVT = np.dot(v.T, v)
        XX = VVT * self.params.b + np.eye(self.params.m_num_factors) * self.params.lambda_u

        item_ids = user_vector
        n = len(item_ids)
        if n > 0:
            A = np.copy(XX)
            A += np.dot(self.m_V[item_ids, :].T, self.m_V[item_ids, :]) * a_minus_b
            x = self.params.a * np.sum(self.m_V[item_ids, :], axis=0)
            return scipy.linalg.solve(self.params.a, x).reshape(1, self.params.n_z)
        else:
            return np.array([])
