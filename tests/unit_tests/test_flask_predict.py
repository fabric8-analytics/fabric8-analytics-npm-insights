"""Test the rest API."""
import unittest
import json

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

    def test_recommendation(self):
        """Test the recommendation endpoint."""
        data = [
            {
                "package_list": [
                  "pon-logger"
                ],
                "comp_package_count_threshold": 0
            }
        ]
        self.recommendation = self.client.post('/api/v1/companion_recommendation',
                                               data=json.dumps(data),
                                               headers={'content-type': 'application/json'})


if __name__ == '__main__':
    unittest.main()
