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
import pandas as pd
from training.datastore.get_keywords import GetKeywords
import unittest

kws_obj = GetKeywords(data_obj=None)

with open('tests/test_data/training-utils/test-node-package-details-with-url.json') as f:
    test_data_df = pd.DataFrame(json.load(f))


class TestGetKeywords(unittest.TestCase):
    """This class tests the Get Keywords Class."""

    def test_from_existing_df(self):
        """Test extraction of data from existing dataframe."""
        test_package = str('algorithm')
        test_data = kws_obj.from_existing_df(test_data_df, test_package)
        assert len(test_data) == 4

    def test_get_version(self):
        """Test getting of latest version."""
        test_api_data = {'dist-tags': {'latest': '1.12.1'}}
        test_version = kws_obj.get_version(test_api_data)
        assert test_version == str('1.12.1')

    def test_get_dependencies(self):
        """Test of getting dependencies."""
        test_api_data = {'dist-tags': {'latest': '1.12.1'}, 'versions':
                         {'1.12.1': {'dependencies': {'a': '1.3', 'b': '1.5'}}}}
        test_dependencies = kws_obj.get_dependencies(test_api_data)
        assert test_dependencies == list(['a', 'b'])

    def test_clean_response(self):
        """Test clean response function."""
        test_topics_list = [{'topic': {'name': 'javascript'}},
                            {'topic': {'name': 'nodejs'}},
                            {'topic': {'name': 'expressjs'}},
                            {'topic': {'name': 'serve-files'}},
                            {'topic': {'name': 'send'}}]
        test_dict_ = {'name': 'serve-static', 'url': 'https://github.com/expressjs/serve-static',
                      'description': 'Serve static files', 'repositoryTopics':
                          {'nodes': test_topics_list}}
        test_response_json = {'data': {'organization': {'name': 'expressjs',
                                                        'url': 'https://github.com/expressjs',
                                                        'repository': test_dict_}}}
        test_topic_name = kws_obj.clean_response(test_response_json)
        assert test_topic_name == ['javascript', 'nodejs', 'expressjs', 'serve-files', 'send']


if __name__ == '__main__':
    unittest.main()
