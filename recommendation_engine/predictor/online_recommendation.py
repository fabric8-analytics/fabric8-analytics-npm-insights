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
from .abstract_recommender import AbstractRecommender
import json
import sys

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

    def _get_new_user_item_vector(self, user_rating_vector):
        """Create this users' m_U vector.

        Create the factor space mapping for this user, this
        can then be multiplied by the latent item space mapping
        to get the recommendations for this user.

        :user_rating_vector: The users' manifest mapped to package vocabulary
        :returns: The 1XD vector for this user, where D is the number of
                  latent factors.
        """
        return

    def _map_package_id_to_name(self, package_id_list):
        """Map the package id from the model to its name.

        :package_id_list: A python iterable containing all the package ids
        """
        return [self.package_id_name_map[str(package_id)] for package_id in package_id_list]

    def _map_package_name_to_id(self, package_name_list):
        return [self.package_name_id_map[package_name] for package_name in package_name_list]

    def _load_package_id_to_name_map(self, path_to_pkg_map):
        """Load the package-id to name mapping.

        :path_to_pkg_map: The URL/URI for the package map.

        """
        with open('/Users/avgupta/analytics-proof-of-concepts/index_to_package_map.json') as idx_pkg_map:
            self.package_id_name_map = json.loads(idx_pkg_map.read())
        with open('/Users/avgupta/analytics-proof-of-concepts/package_to_index_map.json') as pkg_idx_map:
            self.package_name_id_map = json.loads(pkg_idx_map.read())

    def load_rating(self, path=''):
      path = "/Users/avgupta/Dropbox/CVAE/packagedata-train-5-users.dat"
      arr = []
      for line in open(path):
        a = line.strip().split()
        if a[0] == 0:
          l = []
        else:
          l = set([int(x) for x in a[1:]])
        arr.append(l)
      self.user_stacks = arr

    def _find_closest_user_in_training_set(self, new_user_stack):
        new_user_stack = set(new_user_stack)
        minDiff = sys.maxsize
        closest = 0
        print('user stack is: {}'.format(new_user_stack))
        for idx, stack in enumerate(self.user_stacks):
            if (idx == 1239):
                print(stack)
            if stack == new_user_stack:
                closest = idx
                print("Breaking at {}".format(idx))
                break
            elif len(stack.difference(new_user_stack)) < minDiff:
                minDiff = len(stack.difference(new_user_stack))
                closest = idx
                print("setting closest to {}".format(idx))
        print("Closest is: {}".format(closest))
        return closest

    def predict(self, new_user_stack):
        self.load_rating()
        self._load_model_output_matrices('/Users/avgupta/Dropbox/CVAE/cvae.mat')
        self._load_package_id_to_name_map('')
        new_user_stack = self._map_package_name_to_id(new_user_stack)
        user = self._find_closest_user_in_training_set(new_user_stack)
        # print(self.user_matrix[int(user), :].shape)
        recommendation = np.dot(self.user_matrix[int(user), :].reshape([1, 50]), self.latent_item_rep_mat.T)
        packages = np.argsort(recommendation)[0][::-1][:self._M]
        # print(packages)
        return dict(zip(self._map_package_id_to_name(packages.tolist()), [self.sigmoid(rec) for rec in np.sort(recommendation)[0][::-1][:self._M]]))

    def sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

