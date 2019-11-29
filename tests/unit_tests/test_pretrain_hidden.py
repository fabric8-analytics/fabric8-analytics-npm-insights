#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for pretraining hidden file.

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
from recommendation_engine.autoencoder.pretrain import pretrain_hidden
import tensorflow as tf

data = np.load('tests/test_data/2019-01-03/data/content_matrix.npz')
data = data['matrix']
test_shape = np.shape(data)[1]


class TestPretrainHidden(tf.test.TestCase):
    """This class tests Pretrain Hidden class."""

    def test_train(self, test_data=data, test_shape=test_shape):
        """Test train function."""
        test_hidden_obj = pretrain_hidden.PretrainHidden(layer_dim=10)
        test_out = test_hidden_obj.train(data=test_data, input_dimension=test_shape)
        self.assertEqual(np.shape(test_out)[0], 135)
        self.assertEqual(test_hidden_obj.layer_dim, 10)
