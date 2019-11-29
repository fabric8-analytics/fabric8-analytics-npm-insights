#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This file contains the class that defines the online scoring logic.

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
import logging
import sys
import daiquiri
import numpy as np
from recommendation_engine.config.params_scoring import ScoringParams
from recommendation_engine.config.path_constants import PMF_MODEL_PATH, PACKAGE_TAG_MAP, \
    ITEM_USER_FILEPATH, PRECOMPUTED_MANIFEST_PATH, ID_TO_PACKAGE_MAP, PACKAGE_TO_ID_MAP
from recommendation_engine.model.pmf_prediction import PMFScoring
from recommendation_engine.predictor.abstract_recommender import AbstractRecommender
from recommendation_engine.utils.fileutils import load_rating
from recommendation_engine.config.cloud_constants import S3_BUCKET_NAME

daiquiri.setup(level=logging.WARNING)
_logger = daiquiri.getLogger(__name__)


class PMFRecommendation(AbstractRecommender):
    """Online recommendation logic.

    This class contains the online recommendation logic that will be used to
    score packages to the user's preferences at runtime. We need to run a
    single step of PMF and multiply the obtained user vector with the
    precomputed latent item vectors.
    """

    def __init__(self, M, data_store, num_latent=ScoringParams.num_latent_factors):
        """Construct a new instance.

        :M: This parameter controls the number of recommendations that will
            be served by the PMF model.
        """
        AbstractRecommender.__init__(self)
        self._M = M
        self.num_latent = num_latent
        self.user_matrix = None
        self.latent_item_rep_mat = None
        self.weight_matrix = None
        self.s3_client = data_store
        self._load_model_output_matrices(model_path=PMF_MODEL_PATH)
        self._load_package_id_to_name_map()
        self._package_tag_map = self.s3_client.read_json_file(PACKAGE_TAG_MAP)
        self.item_ratings = load_rating(ITEM_USER_FILEPATH, data_store)
        self.user_stacks = load_rating(PRECOMPUTED_MANIFEST_PATH, data_store)
        _logger.info("Created an instance of pmf-recommendation, loaded data from S3")

    def _load_model_output_matrices(self, model_path):
        """Load the m_U, m_V and m_theta matrices.

        :model_path: The path to the matlab file containing the matrices that
                     form the model.
        :returns: An instance of the scoring object.
        """
        _logger.warning("S3 bucket is: {}".format(S3_BUCKET_NAME))
        _logger.warning("Picking model from {}".format(model_path))
        self.model_dict = self.s3_client.load_matlab_multi_matrix(model_path)
        self.user_matrix = self.model_dict["m_U"]
        self.latent_item_rep_mat = self.model_dict["m_V"]
        self.weight_matrix = self.model_dict["m_theta"]

    def _load_package_id_to_name_map(self):
        """Load the package-id to name mapping."""
        _logger.warning("Reading package id map from: {}".format(PACKAGE_TO_ID_MAP))
        self.package_id_name_map = self.s3_client.read_json_file(filename=ID_TO_PACKAGE_MAP)
        self.package_name_id_map = self.s3_client.read_json_file(filename=PACKAGE_TO_ID_MAP)

    def _find_closest_user_in_training_set(self, new_user_stack):
        """Check if we already have the recommendations for the stack precomputed."""
        new_user_stack = set(new_user_stack)
        minDiff = sys.maxsize
        closest = None
        for idx, stack in enumerate(self.user_stacks):
            stack = set(stack)
            diff = len(stack.difference(new_user_stack))
            if stack == new_user_stack:
                closest = idx
                break
            elif new_user_stack.issubset(stack) and diff < minDiff:
                minDiff = diff
                closest = idx
        return closest

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, new_user_stack, companion_threshold=None):
        """Predict companion packages."""
        missing = []
        avail = []
        if not companion_threshold:
            companion_threshold = self._M
        for package in new_user_stack:
            pkg_id = self.package_name_id_map.get(package, -1)
            if pkg_id == -1:
                missing.append(package)
            else:
                avail.append(pkg_id)
        package_topic_dict = {
            self.package_id_name_map[str(package_id)]: self._package_tag_map.get(
                self.package_id_name_map[str(package_id)], []) for package_id in avail
        }
        # if more than half the packages are missing
        if len(avail) == 0 or len(missing) > len(avail):
            return missing, [], package_topic_dict
        new_user_stack = avail
        # Check whether we have already seen this stack.
        user = self._find_closest_user_in_training_set(new_user_stack)
        if user is not None:
            _logger.info("Have precomputed stack")
            recommendation = np.dot(self.user_matrix[int(user), :].reshape([1, self.num_latent]),
                                    self.latent_item_rep_mat.T)
        else:
            _logger.info("Calculating latent representation, have not seen this combination before")
            scoring = PMFScoring(self.model_dict, self.item_ratings)
            user_latent_rep = scoring.predict_transform(new_user_stack, self.num_latent)
            recommendation = np.dot(user_latent_rep.reshape([1, self.num_latent]),
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
            if recommendation_count >= companion_threshold:
                break
        logits = np.take(recommendation[0], np.array(packages_filtered)).tolist()
        mean = np.mean(logits)
        recommendations = []
        for idx, package in enumerate(packages_filtered):
            recommendation = {
                "package_name": self.package_id_name_map[str(package)],
                "cooccurrence_probability": self._sigmoid(logits[idx] - mean) * 100,
                "topic_list": self._package_tag_map.get(
                        self.package_id_name_map[str(package)], [])
            }
            if recommendation.get('cooccurrence_probability') >= ScoringParams.min_confidence_prob:
                recommendations.append(recommendation)
        return missing, recommendations, package_topic_dict
