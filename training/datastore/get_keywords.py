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
from training.datastore.utils import Utility
import pandas as pd
import requests
import json


class GetKeywords:
    """This class defines the S3 Connections viz fetching and storing data."""

    def __init__(self, data_obj, df_=pd.DataFrame(), dict_=dict()):
        """Create an instance for GetKeywords."""
        self.df_ = df_
        self.dict_ = dict_
        self.get_data = data_obj
        self.utility = Utility()

    def from_existing_df(self, df_, package):
        """Find the keywords from existing dump."""
        if not df_.empty:
            data_lst = df_.loc[df_['name'] == str(package),
                               ['name', 'description', 'keywords', 'dependencies']]
            if not data_lst.empty:
                return data_lst.iloc[0]
        else:
            logger.info("Node Package details Dataframe is not existed.")
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
            data_dict['dependencies'] = self.get_dependencies(json_data)
            return data_dict
        except Exception:
            logger.error("Can't fetch the keywords from NPM Registry")
            return data_dict

    def get_version(self, api_data):
        """Give the latest version for the package."""
        if api_data:
            try:
                latest_version = api_data['dist-tags']['latest']
                return latest_version
            except Exception:
                logger.info("Unable to fetch latest version from API data.")
                return ''
        else:
            logger.error("API Data is not available.")
            return ''

    def get_dependencies(self, api_data):
        """Give the dependencies for latest version of package."""
        version = self.get_version(api_data)
        logger.info("Latest_version is: {}".format(version))
        versions_dict = api_data.get('versions', dict())
        try:
            if versions_dict:
                latest_version_data_dict = versions_dict.get(version, dict())
                if latest_version_data_dict:
                    latest_dependencies = latest_version_data_dict.get('dependencies',
                                                                       list())
                    return list(latest_dependencies.keys())
        except Exception:
            return list()

    def clean_response(self, response_json):
        """Clean the api response json."""
        topic_lst = response_json['data']['organization']['repository']['repositoryTopics']['nodes']
        topic_name_lst = [dict(i.get('topic')).get('name') for i in topic_lst]
        return list(topic_name_lst)

    def from_github(self, package, url_df, api_url, api_token):
        """Find the keywords from the Github Graph QL."""
        url_ = self.utility.get_url(url_df, package)
        keywords = list()
        if type(url_) == str:
            query_params = self.utility.get_query_params(url_)
            logger.info("Query Parameters are: {}, {}".format(query_params[0], query_params[1]))
            json = {
                'query': '{{organization(login: "{0}"){{name url repository(name: "{1}")\
                {{name url description repositoryTopics(first: 10){{nodes{{topic {{name}}}}}}}}}}}}'
                .format(str(query_params[0]), str(query_params[1]))}
            headers = {'Authorization': 'token %s' % api_token}
            try:
                response = requests.post(url=api_url, json=json, headers=headers)
                keywords = list(self.clean_response(response.json()))
                return keywords
            except Exception:
                logger.error("Either Github token is not present or response is not coming.")
                return keywords
        else:
            return keywords

    def find_keywords(self, data_, list_):
        """Find the keywords for given list of list of raw data."""
        package_lst = self.utility.flatten_list(list_)
        out_lst = list()
        total = len(package_lst)
        index = 0
        for i in package_lst:
            index += 1
            logger.info(f'Processing [{index}/{total}] => package {i}')
            pkg_kwd_lst = data_.get(i, None)
            logger.info(f'Package {i} => {pkg_kwd_lst}')
            if not pkg_kwd_lst or type(pkg_kwd_lst[2]) != list or type(pkg_kwd_lst[3]) != list:
                logger.warn(f'Package {i}, information missing ignoring it')
                pkg_kwd_lst = [i, '', [], []]
            out_lst.append(pkg_kwd_lst)
        return pd.DataFrame(out_lst, columns=['name', 'description', 'keywords', 'dependencies'])
