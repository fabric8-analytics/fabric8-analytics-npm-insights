"""Test that all constants can be imported."""
import unittest
from inspect import getmembers

import recommendation_engine.config.cloud_constants as cloud_constants
import recommendation_engine.config.path_constants as path_constants


class TestImport(unittest.TestCase):
    """Test if import goes well."""

    def test_cloud_constants(self):
        """Test if able to import cloud constants."""
        self.assertTrue(getmembers(cloud_constants))

    def test_path_constants(self):
        """Test if able to import path constants."""
        self.assertTrue(getmembers(path_constants))
