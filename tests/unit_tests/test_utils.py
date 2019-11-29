#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for training utils file.

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

import json
import unittest
from training.datastore.utils import Utility
import pandas as pd

utils_obj = Utility()

with open('tests/test_data/training-utils/test-node-package-details-with-url.json') as f:
    test_data_df = pd.DataFrame(json.load(f))


class TestUtility(unittest.TestCase):
    """This class tests the Utility Class."""

    def test_flatten_list(self):
        """Test Flatten list function."""
        test_list_ = [[1, 2], [3, 4]]
        test_flatten_list_output = utils_obj.flatten_list(test_list_)
        assert type(test_flatten_list_output[0]) != list

    def test_make_list_from_series(self):
        """Test make list from series function."""
        test_series = pd.Series()
        test_series_list = utils_obj.make_list_from_series(test_series)
        assert type(test_series_list) == list

    def test_dict_to_list(self):
        """Test Dictionary to list conversion."""
        test_dict = dict()
        test_dict_list = utils_obj.dict_to_list(test_dict)
        assert type(test_dict_list) == list

    def test_make_manifest_df(self):
        """Test making of manifest Dataframe function."""
        manifest_df, row_count = utils_obj.make_manifest_df(test_data_df, 0)
        assert isinstance(manifest_df, pd.DataFrame)
        assert row_count == 10

    def test_make_filtered_pkg_kwd_df(self):
        """Test making of filtered package keyword dataframe."""
        test_filtered_pkg_kwd_df = utils_obj.make_filtered_pkg_kwd_df(test_data_df, 0)
        assert isinstance(test_filtered_pkg_kwd_df, pd.DataFrame)

    def test_clean_set(self):
        """Test clean set function."""
        test_set = set(['express js'])
        clean_test_set = list(utils_obj.clean_set(test_set))
        assert clean_test_set[0] == "express-js"

    def test_create_pkg_tag_map(self):
        """Test making of package tag map."""
        test_pkg_tag_map, test_vocab = utils_obj.create_pkg_tag_map(pd.DataFrame())
        assert type(test_pkg_tag_map) == dict and type(test_vocab) == set

    def test_create_pkg_dep_map(self):
        """Test making of package dependency map."""
        test_pkg_dep_map, test_first_deps = utils_obj.create_pkg_dep_map(pd.DataFrame())
        assert type(test_pkg_dep_map) == dict and type(test_first_deps) == set

    def test_create_package_map(self):
        """Test create package map function."""
        test_package_list = list(['expressjs', 'npm', 'chai'])
        test_package_map = utils_obj.create_package_map(test_package_list)
        assert test_package_list == list(test_package_map.keys())

    def test_create_vocabulary_map(self):
        """Test create vocabulary map function."""
        test_vocab = list(['template', 'node', 'promises'])
        test_vocab_map = utils_obj.create_vocabulary_map(test_vocab)
        assert test_vocab == list(test_vocab_map.keys())

    def test_make_kwd_dependencies_df(self):
        """Test making of keyword dependency dataframe."""
        test_unique_packages = list(['npm', 'chai', 'FreshDocs'])
        test_keywords_df, test_dependencies_df = utils_obj.make_kwd_dependencies_df(
            test_data_df, test_unique_packages)
        assert isinstance(test_keywords_df, pd.DataFrame) and isinstance(
            test_dependencies_df, pd.DataFrame)

    def test_extract_package_manifest_lst(self):
        """Test exctract package manifest list function."""
        test_packages, test_manifest_list = utils_obj.extract_package_manifest_lst(test_data_df)
        assert type(test_packages) == set and type(test_manifest_list) == list

    def test_get_url(self):
        """Test get url function."""
        test_package_name = str("Reston")
        test_url = utils_obj.get_url(test_data_df, test_package_name)
        assert str(test_url) == str('git://github.com/maxpert/Reston.git')

    def test_get_query_params(self):
        """Test extraction of query parameters."""
        test_repo_url = str('git://github.com/maxpert/Reston.git')
        params = utils_obj.get_query_params(test_repo_url)
        assert params == list(['maxpert', 'Reston.git'])

    def test_make_user_data(self):
        """Test of making user data function."""
        test_manifest_list = [['npm', 'express'], ['chai', 'serve-static']]
        test_packages = list()
        test_user_data = utils_obj.make_user_data(test_manifest_list, test_packages)
        assert type(test_user_data) == list


if __name__ == '__main__':
    unittest.main()
