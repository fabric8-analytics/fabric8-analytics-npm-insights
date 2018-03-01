#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains the class that defines the online scoring logic.

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
import numpy as np
from scipy.io import loadmat
import daiquiri

daiquiri.setup()


class PMFRecommendation(AbstractRecommender):

    """Online recommendation logic.

    This class contains the online recommendation logic that will be used to
    score packages to the user's preferences at runtime. We need to run a
    forward pass of PMF and multiply the obtained user vector with the
    precomputed latent item vectors."""

    def __init__(self, M):
        """Constructor a new instance.

        :M: This parameter controls the number of recommendations that will
            be served by the PMF model.
        """
        AbstractRecommender.__init__(self)
        self._M = M
        self.user_matrix = None
        self.latent_item_rep_mat = None
        self.weight_matrix = None
        # Initialize a logger for this class.
        self._logger = daiquiri.getLogger(self.__class__.__name__)

    def _load_model_output_matrices(self, model_path):
        """This method is used to load the m_U, m_V and m_theta matrices.

        :model_path: The path to the matlab file containing the matrices that
                     form the model.
        :returns: An instance of the scoring object.
        """
        model_dict = loadmat(model_path)
        if not model_dict:
            self._logger.error("Unable to load the model for scoring")
        self.user_matrix = model_dict["m_U"]
        self.latent_item_rep_mat = model_dict["m_V"]
        self.weight_matrix = model_dict["m_theta"]
