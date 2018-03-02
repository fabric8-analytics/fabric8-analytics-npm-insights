#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, json
from predictor.online_recommendation import PMFRecommendation

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return json.dumps({"Status": "ok"})
    else:
        print(request.json)
        rec = PMFRecommendation(10)
        return json.dumps(rec.predict(request.json['stack']))

if __name__ == "__main__":
    app.run(debug=True, port=3000)
