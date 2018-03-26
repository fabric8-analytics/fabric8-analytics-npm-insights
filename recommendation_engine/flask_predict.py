#!/usr/bin/env python
# -*- coding: utf-8 -*-
import flask
from flask import Flask, request, json
from predictor.online_recommendation import PMFRecommendation

app = Flask(__name__)

rec = PMFRecommendation(10)


@app.route('/api/v1/liveness', methods=['GET'])
def liveness():
    return flask.jsonify({}), 200


@app.route('/api/v1/readiness', methods=['GET'])
def readiness():
    return flask.jsonify({"status": "ready"}), 200


@app.route('/api/v1/companion_recommendation', methods=['POST'])
def recommendation():
    global rec
    recommendations = rec.predict(request.json['stack'])
    if recommendations:
        return flask.jsonify(recommendations), 200
    else:
        return flask.jsonify({"status": "error",
                              "response": "Could not recommend packages"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=3000)
