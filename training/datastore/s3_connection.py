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
from rudra.data_store.aws import AmazonS3
from rudra import logger
import pandas as pd
import os


class GetData:
    """This class defines the S3 Connections viz fetching and storing data."""

    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None,
                 aws_bucket_name='cvae-insights',
                 deployment_prefix='dev', model_version='2019-02-27'):
        """Create an instance of GetData."""
        self.aws_access_key_id = os.environ.get('AWS_S3_ACCESS_KEY_ID', '')
        self.aws_secret_access_key = os.environ.get('AWS_S3_SECRET_ACCESS_KEY',
                                                    '')
        self.bucket_name = aws_bucket_name
        self.deployment_prefix = deployment_prefix
        self.version_name = model_version
        self.s3_object = AmazonS3(bucket_name=self.bucket_name,
                                  aws_access_key_id=self.aws_access_key_id,
                                  aws_secret_access_key=self.aws_secret_access_key)
        self.s3_client = self.load_S3()

    def load_S3(self):
        """Establish the connection with S3."""
        self.s3_object.connect()
        if self.s3_object.is_connected():
            logger.info("S3 connection established.")
            return self.s3_object
        else:
            raise Exception

    def load_raw_data(self):
        """Load the raw data from S3 bucket."""
        NPM_raw_data_path = os.path.join("npm", self.deployment_prefix, self.version_name,
                                         "data/manifest.json")
        if (self.s3_client.object_exists(NPM_raw_data_path)):
            try:
                raw_data_dict_ = self.s3_client.read_json_file(NPM_raw_data_path)
                print("Size of Raw Manifest file is: {}".format(len(raw_data_dict_)))
                return raw_data_dict_
            except Exception:
                raise Exception

    def load_existing_data(self):
        """Load the node registry dump from S3 bucket."""
        NPM_clean_json_data_path = os.path.join("npm", "dev", "2019-01-03",
                                                "data/node-package-details-clean.json")
        print(NPM_clean_json_data_path)
        if (self.s3_client.object_exists(NPM_clean_json_data_path)):
            try:
                print("Path Existed")
                existing_data = self.s3_client.read_generic_file(NPM_clean_json_data_path)
                existing_df = pd.read_json(existing_data, lines=True)
                print("Size of Raw Manifest file is: {}".format(len(existing_df)))
                return existing_df
            except Exception:
                raise Exception("S3 connection error")
        else:
            raise ValueError("Given Path is not present.")
