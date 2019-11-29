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
from recommendation_engine.config.path_constants import TEMPORARY_PATH, TEMPORARY_DATA_PATH
from training.datastore.s3_connection import GetData
from training.datastore.get_keywords import GetKeywords
from training.datastore.preprocess_data import PreprocessData
from training.datastore.utils import Utility
from rudra import logger


class GetPreprocessData:
    """This class processes raw data and converts into the input data for models."""

    def __init__(self,
                 aws_access_key_id='',
                 aws_secret_access_key='',
                 aws_bucket_name='cvae-insights',
                 model_version='',
                 num_train_per_user=5):
        """Create an instance for GetPreprocessData."""
        self.obj_ = GetData(aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key,
                            aws_bucket_name=aws_bucket_name,
                            model_version=model_version,
                            num_train_per_user=num_train_per_user)
        self.keyword_obj_ = GetKeywords(self.obj_)
        self.preprocess_data_obj = PreprocessData(data_obj=self.obj_)
        self.utils = Utility()
        self.num_users = num_train_per_user

    def preprocess_data(self):
        """Preprocesses the data and save into temporary storage."""
        package_tag_map, vocabulary, manifest_user_data, unique_packages = \
            self.preprocess_data_obj.update_pkg_tag_map()
        package_tag_map = {k: list(v) for k, v in package_tag_map.items()}
        self.obj_.save_manifest_file_temporary(manifest_user_data,
                                               'manifest_user_data.dat', TEMPORARY_DATA_PATH)
        package_id_map = self.utils.create_package_map(unique_packages)
        id_package_map = dict(zip(range(len(unique_packages)), list(unique_packages)))
        user_train_data, item_train_data, user_test_data, item_test_data = \
            self.obj_.train_test_data()
        content_matrix = self.utils.create_content_matrix(package_tag_map,
                                                          unique_packages, vocabulary)
        self.obj_.save_json_file_temporary(package_id_map, 'package_to_index_map.json',
                                           TEMPORARY_PATH)
        self.obj_.save_json_file_temporary(id_package_map, 'index_to_package_map.json',
                                           TEMPORARY_PATH)
        self.obj_.save_json_file_temporary(package_tag_map, 'package_tag_map.json',
                                           TEMPORARY_PATH)
        self.obj_.save_file_temporary(user_train_data,
                                      "packagedata-train-" + str(self.num_users) + "-users.dat",
                                      TEMPORARY_DATA_PATH)
        self.obj_.save_file_temporary(user_test_data,
                                      "packagedata-test-" + str(self.num_users) + "-users.dat",
                                      TEMPORARY_DATA_PATH)
        self.obj_.save_file_temporary(item_train_data,
                                      "packagedata-train-" + str(self.num_users) + "-items.dat",
                                      TEMPORARY_DATA_PATH)
        self.obj_.save_file_temporary(item_test_data,
                                      "packagedata-test-" + str(self.num_users) + "-items.dat",
                                      TEMPORARY_DATA_PATH)
        self.obj_.save_numpy_matrix_temporary(content_matrix,
                                              'content_matrix.npz',
                                              TEMPORARY_DATA_PATH)
        logger.info("All items are saved successfully in temporary location.")
