#!/bin/bash

gunicorn -b 0.0.0.0:$SERVICE_PORT --workers=2 -k sync -t $SERVICE_TIMEOUT recommendation_engine.flask_predict:app --log-level $FLASK_LOGGING_LEVEL
