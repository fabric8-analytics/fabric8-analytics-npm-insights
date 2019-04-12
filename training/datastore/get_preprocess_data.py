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


obj_ = GetData()
keyword_obj_ = GetKeywords()
preprocess_data_obj = PreprocessData()
raw_data = obj_.load_raw_data()
packages = raw_data.get('package_list', [])
# existing_df = obj_.load_existing_data()
# keyword_df = keyword_obj_.find_keywords(existing_df, packages)
# print(keyword_df)
package_tag_map, vocabulary = preprocess_data_obj.update_pkg_tag_map()
print(package_tag_map, vocabulary)
