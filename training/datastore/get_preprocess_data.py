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
from training.datastore.s3_connection import GetData
from training.datastore.get_keywords import GetKeywords
from training.datastore.preprocess_data import PreprocessData
from training.datastore.utils import Utility
from rudra import logger


class GetPreprocessData:
    """This class processes raw data and converts into the input data for models."""

    def __init__(self,
                 aws_access_key_id='I92KJ1A4NBOTDNII0UE3',
                 aws_secret_access_key='2+Po/7pyFzSNnHFfAQDbLpZ6p/0uHE5Ujx7mE1W7',
                 aws_bucket_name='cvae-insights',
                 model_version='2019-02-27',
                 num_train_per_user=5):

        self.obj_ = GetData(aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key,
                            aws_bucket_name=aws_bucket_name,
                            model_version=model_version,
                            num_train_per_user=num_train_per_user)
        print(type(self.obj_))
        self.keyword_obj_ = GetKeywords(self.obj_)
        # print
        self.preprocess_data_obj = PreprocessData(data_obj=self.obj_)
        self.utils = Utility()

    def preprocess_data(self):
        package_tag_map, vocabulary, manifest_user_data, unique_packages = self.preprocess_data_obj.update_pkg_tag_map()
        package_tag_map = {k: list(v) for k, v in package_tag_map.items()}
        self.obj_.save_manifest_file_temporary(manifest_user_data, 'manifest_user_data.dat', '/tmp/trained-model/')
        package_id_map = self.utils.create_package_map(unique_packages)
        id_package_map = dict(zip(range(len(unique_packages)), list(unique_packages)))
        user_train_data, item_train_data, user_test_data, item_test_data = self.obj_.train_test_data()
        content_matrix = self.utils.create_content_matrix(package_tag_map, unique_packages, vocabulary)
        self.obj_.save_json_file_temporary(package_id_map, 'package_to_index_map.json',
                                           '/tmp/trained-model/')
        self.obj_.save_json_file_temporary(id_package_map, 'index_to_package_map.json',
                                           '/tmp/trained-model/')
        self.obj_.save_json_file_temporary(package_tag_map, 'package_tag_map.json',
                                           '/tmp/trained-model/')
        self.obj_.save_file_temporary(user_train_data, 'user_train_data.dat', '/tmp/trained-model/')
        self.obj_.save_file_temporary(user_test_data, 'user_test_data.dat', '/tmp/trained-model/')
        self.obj_.save_file_temporary(item_train_data, 'item_train_data.dat', '/tmp/trained-model/')
        self.obj_.save_file_temporary(item_test_data, 'item_test_data.dat', '/tmp/trained-model/')
        self.obj_.save_numpy_matrix_temporary(content_matrix, 'content_matrix.npy', '/tmp/trained-model/')
        logger.info("All items are saved successfully in temporary location.")




