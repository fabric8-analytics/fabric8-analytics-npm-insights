#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactions with local filesystem, useful for unit testing.

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
import fnmatch
import json
import logging
import os
import daiquiri
from scipy.io import loadmat

from recommendation_engine.data_store.abstract_data_store import AbstractDataStore

daiquiri.setup(level=logging.WARNING)
_logger = daiquiri.getLogger(__name__)


class LocalFileSystem(AbstractDataStore):
    """Wrapper on local filesystem, API similar to s3DataStore."""

    def __init__(self, src_dir):
        """Create a new local filesystem instance.

        :src_dir: The root directory for local filesystem object
        """
        self.src_dir = src_dir

    def get_name(self):
        """Return name of local filesystem root dir."""
        return "Local filesytem dir: " + self.src_dir

    def list_files(self, prefix=None, max_count=None):
        """List all the json files in the source directory."""
        list_filenames = []
        for root, dirs, files in os.walk(self.src_dir):
            for basename in files:
                if fnmatch.fnmatch(basename, "*.json"):
                    filename = os.path.join(root, basename)
                    filename = filename[len(self.src_dir):]
                    list_filenames.append(filename)
        list_filenames.sort()
        return list_filenames

    def remove_json_file(self, filename):
        """Remove JSON file from the data_input source file path."""
        return os.remove(os.path.join(self.src_dir, filename))

    def read_generic_file(self, filename):
        """Read a file and return its contents."""
        with open(os.path.join(self.src_dir, filename)) as fileObj:
            return fileObj.read()

    def read_json_file(self, filename):
        """Read JSON file from the data_input source."""
        with open(os.path.join(self.src_dir, filename)) as json_fileobj:
            return json.load(json_fileobj)

    def read_all_json_files(self):
        """Read all the files from the data_input source."""
        list_filenames = self.list_files(prefix=None)
        list_contents = []
        for file_name in list_filenames:
            contents = self.read_json_file(filename=file_name)
            list_contents.append((file_name, contents))
        return list_contents

    def write_json_file(self, filename, contents):
        """Write JSON file into data_input source."""
        with open(os.path.join(self.src_dir, filename), 'w') as outfile:
            json.dump(contents, outfile)
        return None

    def upload_file(self, src, target):
        """Upload file into data store."""
        return None

    def download_file(self, src, target):
        """Download file from data store."""
        return None

    def load_matlab_multi_matrix(self, local_filename):
        """Load a '.mat'file & return a dict representation.

        :local_filename: The path of the object.
        :returns: A dict containing numpy matrices against the keys of the
                  multi-matrix.
        """
        model_dict = loadmat(os.path.join(self.src_dir, local_filename))
        if not model_dict:
            _logger.error("Unable to load the model for scoring")
        return model_dict
