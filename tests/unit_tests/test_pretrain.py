#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for pretraining file.

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

import numpy as np
from recommendation_engine.autoencoder.pretrain import pretrain


test_data = np.load('tests/test_data/2019-01-03/data/content_matrix.npz')
test_data = test_data['matrix']


def test_fit(data=test_data):
    """Tests pretrain function."""
    pretrain.fit(data)
    assert True
