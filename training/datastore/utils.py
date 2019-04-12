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
        pkg_lst.append(series_['name'])
        pkg_lst.append(series_['description'])
        pkg_lst.append(series_['keywords'])
        pkg_lst.append(series_['dependencies'])
        return pkg_lst

    def dict_to_list(self, dict_):
        """Create a list from dictionary."""
        return [value for key, value in dict_.items()]

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
                   set_ if word.strip() is not '')

    def create_pkg_tag_map(self, keywords_df):
        """Create a Package Tag map, and Vocabulary."""
        package_tag_map = self.dict_
        vocabulary = self.set_
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
        for k, g in dependencies_df.groupby("name"):
            try:
                package_dep_map[k] = self.clean_set(
                    package_dep_map.get(k, set()).union(set(g["dependencies"].tolist()[0])))
                all_first_lv_deps = all_first_lv_deps.union(set(package_dep_map[k]))
            except Exception:
                pass
        return package_dep_map, all_first_lv_deps

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
        #         list_of_manifest_list = []
        for idx, row in data.iterrows():
            all_packages = all_packages.union(set(dependency.lower()
                                                  for dependency in row['dependencies']))
        #             list_of_manifest_list.append(list(clean_dependencies(row['dependencies'])))
        return all_packages

    def create_content_matrix(self, pkg_tag_map, all_packages, vocabulary, tag_idx_map):
        """Create Content Matrix."""
        pkg_tag_map = self.dict_
        all_packages = self.set_
        vocabulary = self.set_
        tag_idx_map = self.dict_
        content_matrix = np.zeros([len(all_packages), len(vocabulary)])
        if tag_idx_map:
            for idx, package in enumerate(all_packages):
                package_tags = [tag_idx_map[tag] for tag in pkg_tag_map[package]]
                if idx == 0:
                    print("Setting to 1: {}".format(package_tags))
                content_matrix[idx, package_tags] = 1

        return content_matrix
