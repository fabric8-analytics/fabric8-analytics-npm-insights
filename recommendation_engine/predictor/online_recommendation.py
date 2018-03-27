#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file contains the class that defines the online scoring logic.

# Copyright © 2018 Avishkar Gupta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.io import loadmat
import daiquiri
from recommendation_engine.predictor.abstract_recommender import AbstractRecommender
import json
import sys
from recommendation_engine.config.path_constants import *
from recommendation_engine.data_store.s3_data_store import S3DataStore
import recommendation_engine.config.cloud_constants as cloud_constants
from recommendation_engine.utils.fileutils import load_rating
from recommendation_engine.model.pmf_prediction import PMFScoring
daiquiri.setup()
_logger = daiquiri.getLogger(__name__)


class PMFRecommendation(AbstractRecommender):

    """Online recommendation logic.

    This class contains the online recommendation logic that will be used to
    score packages to the user's preferences at runtime. We need to run a
    single step of PMF and multiply the obtained user vector with the
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
        self.s3_client = S3DataStore(src_bucket_name=cloud_constants.S3_BUCKET_NAME,
                                     access_key=cloud_constants.AWS_S3_ACCESS_KEY_ID,
                                     secret_key=cloud_constants.AWS_S3_SECRET_KEY_ID)

        self._load_model_output_matrices(model_path=PMF_MODEL_PATH)
        self._load_package_id_to_name_map()
        self._package_tag_map = self.s3_client.read_json_file(PACKAGE_TAG_MAP)
        self.item_ratings = load_rating(TRAINING_DATA_ITEMS)
        self.user_stacks = load_rating(PRECOMPUTED_STACKS)
        print("Created an instance of pmf-recommendation, loaded data from S3")

    def _load_model_output_matrices(self, model_path):
        """This method is used to load the m_U, m_V and m_theta matrices.

        :model_path: The path to the matlab file containing the matrices that
                     form the model.
        :returns: An instance of the scoring object.
        """
        self.model_dict = self.s3_client.load_matlab_multi_matrix(model_path)
        self.user_matrix = self.model_dict["m_U"]
        self.latent_item_rep_mat = self.model_dict["m_V"]
        self.weight_matrix = self.model_dict["m_theta"]

    def _get_new_user_item_vector(self, user_rating_vector):
        """Create this users' m_U vector.

        Create the factor space mapping for this user, this
        can then be multiplied by the latent item space mapping
        to get the recommendations for this user.

        :user_rating_vector: The users' manifest mapped to package vocabulary
        :returns: The 1XD vector for this user, where D is the number of
                  latent factors.
        """
        return self.VariationalAutoEncoder()

    def _map_package_id_to_name(self, package_id_list):
        """Map the package id from the model to its name.

        :package_id_list: A python iterable containing all the package ids
        """
        return [self.package_id_name_map[str(package_id)] for package_id in package_id_list]

    def _map_package_name_to_id(self, package_name_list):
        """Map the package name to its id."""
        return [self.package_name_id_map[package_name] for package_name in package_name_list]

    def _load_package_id_to_name_map(self):
        """Load the package-id to name mapping."""
        self.package_id_name_map = self.s3_client.read_json_file(filename=ID_TO_PACKAGE_MAP)
        self.package_name_id_map = self.s3_client.read_json_file(filename=PACKAGE_TO_ID_MAP)

    def _find_closest_user_in_training_set(self, new_user_stack):
        """Check if we already have the recommendations for the stack precomputed"""
        new_user_stack = set(new_user_stack)
        # minDiff = sys.maxsize
        closest = None
        for idx, stack in enumerate(self.user_stacks):
           if stack == new_user_stack:
                closest = idx
                break
            # elif len(stack.difference(new_user_stack)) < minDiff:
                # minDiff = len(stack.difference(new_user_stack))
                # closest = idx
        return closest

    def _sigmoid(self, x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def predict(self, new_user_stack):
        """The main prediction function."""
        missing = []
        avail = []
        for package in new_user_stack:
            pkg_id = self.package_name_id_map.get(package, -1)
            if pkg_id == -1:
                missing.append(package)
            else:
                avail.append(pkg_id)
        # if more than half the packages are missing
        if len(missing) >= len(avail):
            return missing, []
        new_user_stack = avail
        # Check whether we have already seen this stack.
        user = self._find_closest_user_in_training_set(new_user_stack)
        if user is not None:
            _logger.info("Have precomputed stack")
            recommendation = np.dot(self.user_matrix[int(user), :].reshape([1, 50]),
                                    self.latent_item_rep_mat.T)
        else:
            _logger.info("Calculating latent representation, have not seen this combination before")
            scoring = PMFScoring(self.model_dict, self.item_ratings)
            user_latent_rep = scoring.predict_transform(new_user_stack)
            recommendation = np.dot(user_latent_rep.reshape([1, 50]),
                                    self.latent_item_rep_mat.T)
        packages = np.argsort(recommendation)[0][::-1].tolist()
        # Filter packages that are present in the input
        packages_filtered = []
        user_stack_lookup = set(new_user_stack)
        recommendation_count = 0
        for package in packages:
            if package not in user_stack_lookup:
                packages_filtered.append(package)
                recommendation_count += 1
            if recommendation_count >= self._M:
                break
        logits = np.take(recommendation[0], np.array(packages_filtered)).tolist()
        mean = np.mean(logits)
        # return dict(zip(self._map_package_id_to_name(packages),
                    # [self._sigmoid(rec - mean) for rec in logits]))
        recommendations = []
        for idx, package in enumerate(packages_filtered):
            recommendations.append({
                    "package": self.package_id_name_map[str(package)],
                    "companion_probability": self._sigmoid(logits[idx] - mean),
                    "tags" : self._package_tag_map.get(self.package_id_name_map[str(package)], [])
                })
        return missing, recommendations
