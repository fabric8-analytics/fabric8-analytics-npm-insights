#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for pretrain latent file.

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
import tensorflow as tf
from recommendation_engine.autoencoder.pretrain import pretrain_latent


class TestPretrainLatent(tf.test.TestCase):
    """This class tests PretrainLatent class."""

    test_data = np.load('tests/test_data/2019-01-03/data/content_matrix.npz')
    test_data = test_data['matrix']

    def test_train(self, data=test_data):
        """Test train function."""
        self.latent_obj = pretrain_latent.PretrainLatent()
        self.latent_obj.train(data)
        self.assertEqual(np.shape(self.latent_obj.weights), (2, 2))
        self.assertEqual(np.shape(self.latent_obj.de_weights), (1, 2))
