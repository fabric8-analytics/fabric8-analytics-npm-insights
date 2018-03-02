#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, json
from .online_recommendation import PMFRecommendation

app = Flask()

@app.route('/')
def predict():
    rec = PMFRecommendation()
    rec.predict(json.loads(request.json))

if __name__ == "__main__":
    app.run()
