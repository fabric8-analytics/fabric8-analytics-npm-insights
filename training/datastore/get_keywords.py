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
from rudra import logger
from training.datastore.s3_connection import GetData
from training.datastore.utils import Utility
import pandas as pd
import requests
import json


class GetKeywords:
    """This class defines the S3 Connections viz fetching and storing data."""

    def __init__(self, df_=pd.DataFrame(), dict_=dict()):
        """Create an instance for GetKeywords."""
        self.df_ = df_
        self.dict_ = dict_
        self.get_data = GetData()
        self.utility = Utility()

    def from_existing_df(self, df_, package):
        """Find the keywords from existing dump."""
        if not df_.empty:
            data_lst = df_.loc[df_['name'] == str(package),
                               ['name', 'description', 'keywords', 'dependencies']].iloc[0]
            return data_lst
        else:
            logger.error("Node Package details Dataframe is not existed.")
            return self.df_

    def from_npm_registry(self, package):
        """Find the keywords from NPM registry(through api)."""
        data_dict = self.dict_
        api_url = "https://registry.npmjs.org/" + str(package)
        try:
            api_data = requests.get(api_url).text
            json_data = json.loads(api_data)
            data_dict['name'] = json_data.get('name', '')
            data_dict['description'] = json_data.get('description', '')
            data_dict['keywords'] = json_data.get('keywords', [])
            return data_dict
        except Exception:
            logger.error("Cants fetch the keywords from NPM Registry")
            return data_dict

    def find_keywords(self, df_, list_):
        """Find the keywords for given list of list of raw data."""
        package_lst = self.utility.flatten_list(list_)
        out_lst = list()
        for i in package_lst:
            #         print(i)
            pkg_kwd_lst = list()
            pkg_kwd_lst = self.utility.make_list_from_series(
                self.from_existing_df(df_, i))
            #         print(pkg_kwd_lst)
            if type(pkg_kwd_lst[0]) == list:
                #             print("Goes here")
                pkg_kwd_dict = self.from_npm_registry(i)
                #             print(pkg_kwd_dict)
                pkg_kwd_lst = list(pkg_kwd_dict.values())
            out_lst.append(pkg_kwd_lst)
        return pd.DataFrame(out_lst, columns=['name', 'description', 'keywords', 'dependencies'])
