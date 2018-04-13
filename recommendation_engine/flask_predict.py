#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Defines the rest API for the recommender.

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
import os

import flask
from flask import Flask, request
from recommendation_engine.predictor.online_recommendation import PMFRecommendation
from recommendation_engine.data_store.s3_data_store import S3DataStore
import recommendation_engine.config.cloud_constants as cloud_constants
from recommendation_engine.config.params_scoring import ScoringParams

app = Flask(__name__)

s3 = S3DataStore(src_bucket_name=cloud_constants.S3_BUCKET_NAME,
                 access_key=cloud_constants.AWS_S3_ACCESS_KEY_ID,
                 secret_key=cloud_constants.AWS_S3_SECRET_KEY_ID)
# This needs to be global as ~200MB of data is loaded from S3 every time an object of this class
# is instantiated.
recommender = PMFRecommendation(ScoringParams.recommendation_threshold, s3)


@app.route('/api/v1/liveness', methods=['GET'])
def liveness():
    """Define the linveness probe."""
    return flask.jsonify({}), 200


@app.route('/api/v1/readiness', methods=['GET'])
def readiness():
    """Define the readiness probe."""
    return flask.jsonify({"status": "ready"}), 200


@app.route('/api/v1/companion_recommendation', methods=['POST'])
def recommendation():
    """Endpoint to serve recommendations."""
    global recommender
    missing, recommendations = recommender.predict(
            request.json['package_list'],
            companion_threshold=request.json['comp_package_count_threshold'])
    return flask.jsonify({
        "missing_packages": missing,
        "companion_packages": recommendations,
        "ecosystem": os.environ.get("CHESTER_SCORING_REGION")
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=3000)
