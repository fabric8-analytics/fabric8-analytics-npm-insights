import os

USE_CLOUD_SERVICES = os.environ.get("USE_CLOUD_SERVICES", True)
AWS_S3_ACCESS_KEY_ID = os.environ.get('AWS_S3_ACCESS_KEY_ID', '')
AWS_S3_SECRET_KEY_ID = os.environ.get('AWS_S3_SECRET_ACCESS_KEY', '')
DEPLOYMENT_PREFIX = os.environ.get('DEPLOYMENT_PREFIX', 'dev')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'avgupta-stack-analysis-dev')
AWS_S3_ENDPOINT_URL = os.environ.get('AWS_S3_ENDPOINT_URL', '')
