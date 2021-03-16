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
from training.datastore.utils import Utility
from recommendation_engine.config.path_constants import TEMPORARY_DATA_PATH
from rudra import logger
import numpy as np
import time
import os
import json
import io


class GetData:
    """This class defines the S3 Connections viz fetching and storing data."""

    def __init__(self,
                 aws_access_key_id,
                 aws_secret_access_key,
                 num_train_per_user,
                 aws_bucket_name,
                 model_version):
        """Create an instance of GetData."""
        self.aws_access_key_id = os.environ.get('AWS_S3_ACCESS_KEY_ID', aws_access_key_id)
        self.aws_secret_access_key = os.environ.get('AWS_S3_SECRET_ACCESS_KEY',
                                                    aws_secret_access_key)
        self.github_token = os.environ.get('GITHUB_TOKEN', '')
        self.bucket_name = aws_bucket_name
        self.version_name = model_version
        self.s3_object = AmazonS3(bucket_name=self.bucket_name,
                                  aws_access_key_id=self.aws_access_key_id,
                                  aws_secret_access_key=self.aws_secret_access_key
                                  )
        self.num_train_per_user = num_train_per_user
        self.s3_client = self.load_s3()
        self.utility = Utility()

    def load_s3(self):
        """Establish the connection with S3."""
        self.s3_object.connect()
        if self.s3_object.is_connected():
            logger.info("S3 connection established.")
            return self.s3_object
        else:
            raise Exception

    def load_raw_data(self):
        """Load the raw data from S3 bucket."""
        NPM_raw_data_path = os.path.join(self.version_name,
                                         "data/manifest.json")
        logger.info("Reading raw data from {}".format(self.version_name))
        if (self.s3_client.object_exists(NPM_raw_data_path)):
            try:
                raw_data_dict_ = self.s3_client.read_json_file(NPM_raw_data_path)
                logger.info("Size of Raw Manifest file is: {}".format(len(raw_data_dict_)))
                return raw_data_dict_
            except Exception:
                raise Exception

    def load_existing_data(self):
        """Load the node registry dump from S3 bucket."""
        NPM_clean_json_data_path = os.path.join("training-utils",
                                                "node-package-details.json")
        if self.s3_client.object_exists(NPM_clean_json_data_path):
            try:
                logger.info("Reading dump data from training-utils folder.")
                existing_data = self.s3_client.read_json_file(NPM_clean_json_data_path)
                logger.info("Size of raw json: {}".format(len(existing_data)))
                return existing_data
            except Exception:
                raise Exception("S3 connection error")
        else:
            raise ValueError("Given Path is not present.")

    def _read_json_file(self, data_in_bytes):  # pragma: no cover
        """Read a big json file."""
        try:
            coded_data = data_in_bytes.decode('utf-8')
            io_data = io.StringIO(coded_data)
            json_data = io_data.readlines()
            data = list(map(json.loads, json_data))
            return data
        except Exception:
            logger.error("Unable to read json file.")
            return None

    def load_package_data(self):
        """Load the node registry dump from S3 bucket."""
        NPM_clean_json_data_path = os.path.join("training-utils",
                                                "node-package-details-with-url.json")
        if self.s3_client.object_exists(NPM_clean_json_data_path):
            try:
                logger.info("Reading dump data from training-utils folder.")
                existing_data = self.s3_client.read_generic_file(NPM_clean_json_data_path)
                existing_df = self._read_json_file(existing_data)
                logger.info("Size of Raw df with url is: {}".format(len(existing_df)))
                return existing_df
            except Exception:
                raise Exception("S3 connection error")
        else:
            raise ValueError("Given Path is not present.")

    def load_user_item_data(self):
        """Load the manifest file."""
        NPM_manifest_user_data_path = os.path.join(TEMPORARY_DATA_PATH, "manifest_user_data.dat")
        try:
            with open(NPM_manifest_user_data_path, 'rb') as f:
                user_item_data = f.read()
            return user_item_data
        except Exception:
            raise Exception("S3 could not read the file.")

    def create_package_train_user_data(self):
        """Create package train user data."""
        self.package_train_user_data = list()
        for user_id in range(self.num_users):
            this_user_items = self.pairs_train[self.pairs_train[:, 0] == user_id, 1]
            items_str = " ".join(str(x) for x in this_user_items)
            self.package_train_user_data.append([len(this_user_items), items_str])
        return self.package_train_user_data

    def create_package_train_item_data(self):
        """Create package train item data."""
        self.package_train_item_data = list()
        for item_id in range(self.num_items):
            this_item_users = self.pairs_train[self.pairs_train[:, 1] == item_id, 0]
            users_str = " ".join(str(x) for x in this_item_users)
            self.package_train_item_data.append([len(this_item_users), users_str])
        return self.package_train_item_data

    def create_package_test_user_data(self):
        """Create package test user data."""
        self.package_test_user_data = list()
        for user_id in range(self.num_users):
            this_user_items = self.pairs_test[self.pairs_test[:, 0] == user_id, 1]
            items_str = " ".join(str(x) for x in this_user_items)
            self.package_test_user_data.append([len(this_user_items), items_str])
        return self.package_test_user_data

    def create_package_test_item_data(self):
        """Create package test item data."""
        self.package_test_item_data = list()
        for item_id in range(self.num_items):
            this_item_users = self.pairs_test[self.pairs_test[:, 1] == item_id, 0]
            users_str = " ".join(str(x) for x in this_item_users)
            self.package_test_item_data.append([len(this_item_users), users_str])
        return self.package_test_item_data

    def train_test_data(self):
        """Create the training testing data for PMF."""
        data_list = self.split_training_testing_data()
        self.pairs_train = data_list[0]
        self.pairs_test = data_list[1]
        self.num_users = data_list[2]
        self.num_items = data_list[3]
        packagedata_train_users = self.create_package_train_user_data()
        packagedata_train_items = self.create_package_train_item_data()
        packagedata_test_users = self.create_package_test_user_data()
        packagedata_test_items = self.create_package_test_item_data()
        return packagedata_train_users, packagedata_train_items, \
            packagedata_test_users, packagedata_test_items

    def split_training_testing_data(self):
        """Split data into training and testing."""
        data_in_bytes = self.load_user_item_data()
        data = data_in_bytes.decode("utf-8")
        data_list = data.split('\n')
        pairs_train = []
        pairs_test = []
        user_id = 0
        np.random.seed(int(time.time()))
        logger.info("Splitting data into training and testing.")
        for line in data_list:
            arr = line.strip().split()
            arr = np.asarray([int(x) for x in arr[1:]])
            n = len(arr)
            idx = np.random.permutation(n)
            for i in range(min(self.num_train_per_user, n)):
                pairs_train.append((user_id, arr[idx[i]]))
            if n > self.num_train_per_user:
                for i in range(self.num_train_per_user, n):
                    pairs_test.append((user_id, arr[idx[i]]))
            user_id += 1
        num_users = user_id
        pairs_train = np.asarray(pairs_train)
        pairs_test = np.asarray(pairs_test)
        num_items = np.maximum(np.max(pairs_train[:, 1]), np.max(pairs_test[:, 1])) + 1
        logger.info("Number of users and items are respectively {},"
                    " {}".format(num_users, num_items))
        return [pairs_train, pairs_test, num_users, num_items]

    def check_path(self, path):
        """Check the given datastore path."""
        logger.info("Given path is: {}".format(path))
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            return path
        except Exception as e:
            raise e

    def save_file_temporary(self, content, filename, datastore):
        """Store data file in temporary storage."""
        path = self.check_path(datastore)
        try:
            with open(os.path.join(path, filename), 'w') as f:
                for lst in content:
                    ele_str = " ".join([str(x) for x in lst[1:]])
                    f.write("{} {}\n".format(lst[0], ele_str))
            logger.info("File has been stored successfully.")
        except Exception as e:
            raise e

    def save_manifest_file_temporary(self, content, filename, datastore):
        """Store manifest file in temporary storage."""
        path = self.check_path(datastore)
        try:
            with open(os.path.join(path, filename), 'w') as f:
                for lst in content:
                    f.write("{} {}\n".format(lst[0], " ".join(str(x) for x in lst[1:])))
            logger.info("Manifest File has been stored successfully.")

        except Exception as e:
            raise e

    def save_numpy_matrix_temporary(self, content, filename, datastore):
        """Store numpy matrix in temporary storage."""
        path = self.check_path(datastore)
        try:
            np.savez(os.path.join(path, filename), matrix=content)
            logger.info("Numpy matrix has been stored successfully.")

        except Exception as e:
            raise e

    def save_json_file_temporary(self, content, filename, datastore):
        """Store JSON file in temporary storage."""
        path = self.check_path(datastore)
        try:
            with open(os.path.join(path, filename), 'w') as f:
                json.dump(content, f)
            logger.info("JSON file has been stored successfully.")
        except Exception as e:
            raise e

    def save_on_s3(self, folder_path):
        """Store all the contents on S3."""
        try:
            if os.path.exists(folder_path):
                if 'intermediate-model' in folder_path:
                    self.s3_client.s3_upload_folder(folder_path=folder_path,
                                                    prefix=self.version_name + '/intermediate-model'
                                                    )
                else:
                    self.s3_client.s3_upload_folder(folder_path=folder_path,
                                                    prefix=self.version_name + '')
                logger.info("Folders are successfully saved on S3.")
            else:
                logger.error("Folder path doesn't exist.")
        except Exception as e:
            raise e
