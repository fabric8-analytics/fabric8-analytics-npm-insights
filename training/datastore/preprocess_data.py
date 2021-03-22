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
import pandas as pd
from training.datastore.utils import Utility
from training.datastore.get_keywords import GetKeywords


class PreprocessData:
    """This class defines the PreprocessData functions."""

    def __init__(self, data_obj, df_=pd.DataFrame()):
        """Create an instance for PreprocessData."""
        self.get_data_obj = data_obj
        self.utility_obj = Utility()
        self.get_keywords_obj = GetKeywords(data_obj)
        self.df_ = df_
        self.existing_data = self.get_data_obj.load_existing_data()
        self.pkg_kwd_df = self.fetch_package_keywords()

    def add_dependencies_resolved_column(self, df_, dependencies_list):
        """Return a binary value for dependency resoled column."""
        dependencies = [dep.lower() for dep in dependencies_list]
        pkg_with_tags = df_.loc[df_['name'].isin(dependencies)]
        if len(pkg_with_tags) == 0:
            return 0
        elif len(set(dependencies) - set(pkg_with_tags['name'])) == 0:
            return 1
        else:
            return 0

    def check_resolved_dependencies(self, df_):
        """Add a column all dependencies resolved and assign the binary value."""
        df_['all_deps_resolved'] = [self.add_dependencies_resolved_column(self.pkg_kwd_df, i)
                                    for i in df_['dependencies']]
        df_ = df_.loc[df_['all_deps_resolved'] == 0]
        return df_

    def fetch_package_keywords(self):
        """Fetch the keywords for raw data's package list."""
        raw_data = self.get_data_obj.load_raw_data()
        manifest_data = raw_data.get('package_dict', {})
        all_manifest = manifest_data.get('user_input_stack', []) + \
            manifest_data.get('bigquery_data', [])
        try:
            package_keyword_df = self.get_keywords_obj.find_keywords(
                self.existing_data, all_manifest)
            return package_keyword_df
        except Exception:
            raise ValueError("Unable to fetch keywords.")

    def make_necessary_df(self, limit_manifest, limit_keywords):
        """Create two dataframes for dependencies and keywords respectively.."""
        filtered_pkg_kwd_df = self.df_
        manifest_df = self.df_
        if 'dependencies' in self.pkg_kwd_df.columns:
            manifest_df = self.utility_obj.make_manifest_df(self.pkg_kwd_df, limit_manifest)
        else:
            raise KeyError("Dependency is not present")
        if 'keywords' in self.pkg_kwd_df.columns:
            filtered_pkg_kwd_df = self.utility_obj.make_filtered_pkg_kwd_df(
                self.pkg_kwd_df, limit_keywords)
        else:
            raise KeyError("Keywords are not present")

        return (list([manifest_df, filtered_pkg_kwd_df]))

    def extract_unique_packages(self):
        """Return all unique packages from filtered package keyword dataframe."""
        filtered_pkg_kwd_df = self.make_necessary_df(5, 0)[1]
        data_with_dep_check = self.check_resolved_dependencies(filtered_pkg_kwd_df)
        unique_packages, manifest_user_data = self.utility_obj.extract_package_manifest_lst(
            data_with_dep_check)
        manifest_user_data = self.utility_obj.make_user_data(
            manifest_user_data, unique_packages)
        return unique_packages, manifest_user_data

    def create_df_and_dictionaries(self):
        """Create all the necessary dataframes and dictionaries."""
        self.unique_packages, self.manifest_user_data = self.extract_unique_packages()
        self.keyword_df, self.dependencies_df = self.utility_obj.make_kwd_dependencies_df(
            self.pkg_kwd_df, self.unique_packages)
        self.package_tag_map, self.vocabulary = self.utility_obj.create_pkg_tag_map(
            self.keyword_df)
        self.package_dep_map, self.first_level_deps = self.utility_obj.create_pkg_dep_map(
            self.dependencies_df)

    def create_extended_pkg_tag_map(self):
        """Create the package tag map according to all first level dependencies."""
        self.create_df_and_dictionaries()
        self.extended_ptm = dict()
        keywords_df_deps = self.pkg_kwd_df.loc[self.pkg_kwd_df['name'].isin(self.first_level_deps),
                                               ['name', 'keywords']]
        for k, g in keywords_df_deps.groupby("name"):
            try:
                self.extended_ptm[k] = self.utility_obj.clean_set(self.
                                                                  package_dep_map.get(k, set()).
                                                                  union(set(g["keywords"].
                                                                            tolist()[0])))
            except Exception:
                pass

        return self.extended_ptm, self.manifest_user_data, self.unique_packages

    def update_pkg_tag_map(self):
        """Update the existing package tag map."""
        extended_ptm, manifest_user_data, unique_packages = self.create_extended_pkg_tag_map()
        for package_name in self.package_tag_map.keys():
            more_keywords = set()
            for dependency in self.package_dep_map[package_name]:
                more_keywords = more_keywords.union(set(extended_ptm.get(dependency, [])))
            self.package_tag_map[package_name] = self.package_tag_map.get(
                package_name).union(more_keywords)
            self.vocabulary = self.vocabulary.union(more_keywords)
        return self.package_tag_map, self.vocabulary, manifest_user_data, unique_packages
