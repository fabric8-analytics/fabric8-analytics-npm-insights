#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for PMF Recommendation module.

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
from unittest import TestCase
from rudra.data_store.local_data_store import LocalDataStore
from recommendation_engine.predictor.online_recommendation import PMFRecommendation


class TestPMFRecommendation(TestCase):
    """Test the core recommendations task."""

    def setUp(self):
        """Instantiate the resources required for the tests."""
        self.fs = LocalDataStore('tests/test_data')
        self.assertTrue(self.fs.get_name().endswith('tests/test_data'))
        self.pmf_rec = PMFRecommendation(2, data_store=self.fs, num_latent=5)

    def test__find_closest_user_in_training_set(self):
        """Test if we are getting correct "closest user" from the training set."""
        # Full match
        closest = self.pmf_rec._find_closest_user_in_training_set([17190, 14774, 15406, 16594,
                                                                   29063])
        self.assertIsNotNone(closest)
        # Partial
        closest = self.pmf_rec._find_closest_user_in_training_set([17190, 14774, 15406])
        self.assertIsNotNone(closest)
        # Negative
        closest = self.pmf_rec._find_closest_user_in_training_set([3, 4])
        self.assertIsNone(closest)

    def test__sigmoid(self):
        """Test if the sigmoid function is behaving correctly."""
        self.assertEqual(self.pmf_rec._sigmoid(0), 0.5)

    def test_predict(self):
        """Test the prediction flow."""
        # Test for a new stack.
        missing, recommendation, ptm = self.pmf_rec.predict(['pon-logger'])
        self.assertFalse(missing)
        # Should have two recommendations here.
        self.assertEqual(len(recommendation), 2)

        # Tests for missing package.
        missing, recommendation, _ = self.pmf_rec.predict(['pon-logger', 'missing'])
        self.assertTrue(missing)
        # Test if still getting recommendation as no. of missing = no. of known
        self.assertGreater(len(recommendation), 0)

        missing, _, package_tag_map = self.pmf_rec.predict(['missing'])
        self.assertDictEqual(package_tag_map, {})

        # Test for precomputed stack.
        _, recommendation, _ = self.pmf_rec.predict(['async', 'colors', 'request',
                                                     'underscore', 'pkginfo'])
        self.assertTrue(recommendation)
