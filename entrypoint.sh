#!/bin/bash

gunicorn --pythonpath /recommendation_engine -b 0.0.0.0:$SERVICE_PORT --workers=2 -k sync -t $SERVICE_TIMEOUT flask_predict:app
