# Copyright © 2020 Red Hat Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dharmendra G Patel <dhpatel@redhat.com>
#
"""Class to update NPM package details."""
import io
import json
import requests
import logging as logger
from training.datastore.s3_helper import S3Helper


logger.basicConfig(level=logger.INFO)
# logger = logging.getLogger(__file__)

DEV_MODE_ON = False

NPM_PACKAGE_FILE_PATH = "training-utils/node-package-details-with-url.json"
NPM_PACKAGE_FILE_PATH_NEW = "training-utils/node-package-details.json"

NPM_LOCAL_MANIFEST_FILE_PATH = \
    "/home/dhpatel/Downloads/prod-s3-cvae-npm-insights_2020-07-20/data/manifest.json"
NPM_LOCAL_PACKAGE_FILE_PATH = \
    "/home/dhpatel/Downloads/prod-s3-cvae-npm-insights_training-utils" \
    "/node-package-details-with-url.json"
NPM_LOCAL_PACKAGE_FILE_PATH_NEW = \
    "/home/dhpatel/Downloads/prod-s3-cvae-npm-insights_training-utils/node-package-details.json"


class NPMMetadata:
    """NPM metadata fetcher."""

    def __init__(self, s3Helper=S3Helper, github_token=str,
                 bucket_name=str, manifest_data=dict()):
        """Set obect default memebers."""
        self.s3Helper = s3Helper
        self.github_token = github_token
        self.bucket_name = bucket_name
        self.existing_data = self._get_transform_data()

    def update(self):
        """Read and update metadata for all NPM packages in S3."""
        logger.info("Existing node package length: %d", len(self.existing_data))

        self._transform_and_save_data()


    def _get_transform_data(self):
        """Load the node registry dump from S3 bucket and tranform into dict for quick access."""
        if DEV_MODE_ON:
            data = {}
            try:
                with open(NPM_LOCAL_PACKAGE_FILE_PATH, "rb") as fp:
                    coded_data = fp.read().decode("utf-8")
                    io_data = io.StringIO(coded_data)
                    json_data = io_data.readlines()
                    raw_data = list(map(json.loads, json_data))
                    for package in raw_data:
                        package_name = package.get("name", None)
                        if package_name:
                            data[package_name] = package
            except Exception as e:
                logger.warn('Parsing warning raises %s, Trying to read plain json format.', e)
                with open(NPM_LOCAL_PACKAGE_FILE_PATH, "r") as fp:
                    data = json.load(fp)

            return data

        return self.s3Helper.read_json_object(bucket_name=self.bucket_name,
                                            obj_key=NPM_PACKAGE_FILE_PATH) or {}

    def _transform_and_save_data(self):
        """Get back data into original format and save it to a file."""
        if DEV_MODE_ON:
            with open(NPM_LOCAL_PACKAGE_FILE_PATH_NEW, "w+") as fp:
                json.dump(self.existing_data, fp)

        else:
            self.s3Helper.store_json_content(content=self.existing_data,
                                             bucket_name=self.bucket_name,
                                             obj_key=NPM_PACKAGE_FILE_PATH_NEW)
