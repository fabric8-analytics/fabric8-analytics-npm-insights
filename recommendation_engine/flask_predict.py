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
import flask
from flask import Flask, request, json
from recommendation_engine.predictor.online_recommendation import PMFRecommendation

app = Flask(__name__)

rec = PMFRecommendation(10)


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
    global rec
    missing, recommendations = rec.predict(request.json['stack'])
    return flask.jsonify({"missing_packages": missing, "recommendations": recommendations}), 200


if __name__ == "__main__":
    app.run(debug=True, port=3000)
