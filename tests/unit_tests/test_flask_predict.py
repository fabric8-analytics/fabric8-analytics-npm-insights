"""Test the rest API."""
import unittest

import recommendation_engine.config.cloud_constants as cloud_constants
cloud_constants.USE_CLOUD_SERVICES = False
import recommendation_engine.flask_predict as rest_api


class FlaskPredictTestCase(unittest.TestCase):
    """Tests for the API endpoints of the recommender."""

    def setUp(self):
        """Set up fixtures."""
        rest_api.app.testing = True
        rest_api.USE_CLOUD_SERVICES = False
        self.client = rest_api.app.test_client()

    def test_health_checks(self):
        """Test the liveness and readiness probes."""
        self.assertEqual(self.client.get('/api/v1/liveness').status, '200 OK')
        self.assertEqual(self.client.get('/api/v1/readiness').status, '200 OK')


if __name__ == '__main__':
    unittest.main()
