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
import numpy as np
import pandas as pd
import json
import io


class Utility:
    """This class defines the Utility functions."""

    def __init__(self, list_=[], dict_={}, series_=pd.Series(),
                 df_=pd.DataFrame(), set_=set(), limit=0):
        """Create an instance of Utility."""
        self.list_ = list_
        self.dict_ = dict_
        self.series_ = series_
        self.df_ = df_
        self.limit = limit
        self.set_ = set_

    def flatten_list(self, list_):
        """Create a flatten list for given list of list."""
        lst_ = [ele for sublst in list_ for ele in sublst]
        unique_lst_ = list(set(lst_))
        return unique_lst_

    def make_list_from_series(self, series_):
        """Create a list from given series."""
        pkg_lst = list()
        try:
            pkg_lst.append(series_['name'])
            pkg_lst.append(series_['description'])
            pkg_lst.append(series_['keywords'])
            pkg_lst.append(series_['dependencies'])
            return pkg_lst
        except Exception:
            return pkg_lst

    def dict_to_list(self, dict_):
        """Create a list from dictionary."""
        return [value for key, value in dict_.items()]

    def read_json_file(self, data_in_bytes):  # pragma: no cover
        """Read a big json file."""
        try:
            coded_data = data_in_bytes.decode('utf-8')
            io_data = io.StringIO(coded_data)
            json_data = io_data.readlines()
            data = list(map(json.loads, json_data))
            df = pd.DataFrame(data)
            return df
        except Exception:
            logger.error("Unable to read json file.")
            return self.df_

    def make_manifest_df(self, df_, limit):
        """Create a manifest dataframe according to dependencies."""
        df_['id'] = df_.index
        df_['name'] = df_.astype('unicode')['name'].str.lower()
        manifest_df_ = df_[pd.notnull(df_['dependencies'])]
        manifest_df_ = manifest_df_[manifest_df_['dependencies'].str.len() >= int(limit)]
        row_count = manifest_df_.shape[0]
        return manifest_df_, int(row_count)

    def make_filtered_pkg_kwd_df(self, df_, limit):
        """Create a Package keyword dataframe according to keywords."""
        filtered_pkg_kwd_df = pd.DataFrame()
        df_['name'] = df_.astype('unicode')['name'].str.lower()
        try:
            filtered_pkg_kwd_df = df_[pd.notnull(df_['keywords'])]
        except Exception:
            raise Exception("Keywords are not present")
        filtered_pkg_kwd_df = filtered_pkg_kwd_df[filtered_pkg_kwd_df['keywords'].str.len() > limit]
        return filtered_pkg_kwd_df

    def clean_set(self, set_):
        """Clean Keywords and Dependencies."""
        return set(word.strip().lower().replace(' ', '-') for word in
                   set_ if word.strip() != '')

    def create_pkg_tag_map(self, keywords_df):
        """Create a Package Tag map, and Vocabulary."""
        package_tag_map = self.dict_
        vocabulary = self.set_
        if not keywords_df.empty:
            for k, g in keywords_df.groupby("name"):
                try:
                    package_tag_map[k] = self.clean_set(package_tag_map.get(k, set())
                                                        .union(set(g["keywords"].tolist()[0])))
                    vocabulary = vocabulary.union(package_tag_map[k])
                except Exception:
                    pass

        return dict(package_tag_map), set(vocabulary)

    def create_pkg_dep_map(self, dependencies_df):
        """Create a Pakcgae to dependency map and maintain all first level dependencies."""
        package_dep_map = self.dict_
        all_first_lv_deps = self.set_
        if not dependencies_df.empty:
            for k, g in dependencies_df.groupby("name"):
                try:
                    package_dep_map[k] = self.clean_set(
                        package_dep_map.get(k, set()).union(set(g["dependencies"].tolist()[0])))
                    all_first_lv_deps = all_first_lv_deps.union(set(package_dep_map[k]))
                except Exception:
                    pass

        return dict(package_dep_map), set(all_first_lv_deps)

    def create_package_map(self, packages_list):
        """Create Package to index Map."""
        try:
            return dict(zip(list(packages_list), range(len(packages_list))))
        except Exception:
            return dict()

    def create_vocabulary_map(self, vocab_list):
        """Create Vocabulary to Index Map."""
        try:
            return dict(zip(list(vocab_list), range(len(vocab_list))))
        except Exception:
            return dict()

    def make_kwd_dependencies_df(self, data_df, unique_packages):
        """Create Keyword Dependencies Dataframe."""
        keyword_df = self.df_
        dependencies_df = self.df_
        try:
            keyword_df = data_df.loc[data_df['name'].isin(unique_packages), ['name', 'keywords']]
        except Exception:
            logger.error("Keyword is not present.")
        try:
            dependencies_df = data_df.loc[data_df['name'].
                                          isin(unique_packages), ['name', 'dependencies']]
        except Exception:
            logger.error("Dependencies are not present. ")
        return keyword_df, dependencies_df

    def extract_package_manifest_lst(self, data):
        """Extract all unique packages."""
        all_packages = set()
        list_of_manifest_list = []
        for idx, row in data.iterrows():
            all_packages = all_packages.union(set(dependency.lower()
                                                  for dependency in row['dependencies']))
            list_of_manifest_list.append(list(self.clean_set(row['dependencies'])))
        return all_packages, list_of_manifest_list

    def create_content_matrix(self, pkg_tag_map,
                              all_packages, vocabulary, tag_idx_map):  # pragma: no cover
        """Create Content Matrix."""
        pkg_tag_map = self.dict_
        all_packages = self.set_
        vocabulary = self.set_
        tag_idx_map = self.dict_
        content_matrix = np.zeros([len(all_packages), len(vocabulary)])
        if tag_idx_map:
            for idx, package in enumerate(all_packages):
                package_tags = [tag_idx_map[tag] for tag in pkg_tag_map[package]]
                content_matrix[idx, package_tags] = 1

        return content_matrix

    def get_url(self, data_df, pkg_name):
        """Find the package github URL from exisiting DataFrame."""
        try:
            url_ = data_df.loc[data_df['name'] == str(pkg_name), ['repositoryurl']]
            return url_['repositoryurl'].item()
        except Exception:
            return str()

    def get_query_params(self, repo_url):
        """Give the Query Parameters which are organization and package name respectively."""
        org = str()
        package_name = str()
        url_chunks = (repo_url.rsplit('/', 2))
        if 'github' not in url_chunks[1]:
            org = url_chunks[1]
        package_name = url_chunks[2]

        return list([org, package_name])

    def make_user_data(self, manifest_list, unique_packages):
        """Return the user data, which is required for making test data."""
        manifest_user_data = list()
        logger.info("Length of manifest list is: {}".format(len(manifest_list)))
        logger.info("Length of Unique Packages are: {}".format(len(unique_packages)))
        if unique_packages:
            pkg_idx_map = self.create_package_map(unique_packages)
            for manifest in manifest_list:
                this_user_items = [pkg_idx_map[pkg] for pkg in manifest]
                this_user_items = [str(x) for x in this_user_items]
                length_ = len(this_user_items)
                user_items = [str(length_)] + this_user_items
                manifest_user_data.append(user_items)

        return list(manifest_user_data)
