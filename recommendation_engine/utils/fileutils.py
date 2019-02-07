#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains file utilities for loading and writing files to local memory.

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
from recommendation_engine.data_store import data_store_wrapper
"""
import os


def load_rating(path, data_store):
    """Load the rating matrix and return it as a list-of-lists.

    :path: The local pathname for the rating matrix.
    :returns: The rating matrix in a list-of-lists format where the
              list at index i represents the itemeset of the ith user.
    """
    rating_file_contents = data_store.read_generic_file(path)

    if isinstance(rating_file_contents, (bytes, bytearray)):
        rating_file_contents = rating_file_contents.decode('utf-8')

    rating_file_contents = rating_file_contents.strip()

    rating_matrix = []
    for line in rating_file_contents.split('\n'):
        this_user_ratings = line.strip().split()
        if int(this_user_ratings[0]) == 0:
            this_user_item_list = set()
        else:
            this_user_item_list = set([int(x) for x in this_user_ratings[1:]])
        rating_matrix.append(this_user_item_list)
    return rating_matrix


def save_temporary_local_file(buf, local_filename):
    """Save contents of a buffer to local /tmp dir.

    :buffer: The buffer containing the data to write to file.
    :local_filename: The file name of the local file to write to
    :returns: True if success.
    """
    with open(os.path.join('/tmp', local_filename), 'wb') as local_fileobj:
        local_fileobj.write(buf)
