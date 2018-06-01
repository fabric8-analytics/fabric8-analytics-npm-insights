#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Abstract class for data store interactions.

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
import abc


class AbstractDataStore(metaclass=abc.ABCMeta):
    """Abstract class to dictate the behaviour of a data store."""

    @abc.abstractmethod
    def get_name(self):
        """Get name of bucket or root fs directory."""
        raise NotImplementedError()

    @abc.abstractmethod
    def list_files(self, prefix=None, max_count=None):
        """List all the files in the source directory."""
        raise NotImplementedError()

    @abc.abstractmethod
    def read_json_file(self, filename):
        """Read JSON file from the data source."""
        raise NotImplementedError()

    @abc.abstractmethod
    def read_all_json_files(self):
        """Read all the files from the data source."""
        raise NotImplementedError()

    @abc.abstractmethod
    def write_json_file(self, filename, contents):
        """Write JSON file into data source."""
        raise NotImplementedError()

    @abc.abstractmethod
    def upload_file(self, src, target):
        """Upload file into data store."""
        raise NotImplementedError()

    @abc.abstractmethod
    def download_file(self, src, target):
        """Download file from data store."""
        raise NotImplementedError()

    @abc.abstractmethod
    def read_into_file(self, filename):
        """Read from data store and return stream as a file object."""
        raise NotImplementedError()
