"""Various utility functions related to S3 storage."""

import json
import os
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__file__)


class S3Helper:
    """Helper class for storing reports to S3."""

    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, report_bucket=None):
        """Init method for the helper class."""
        self.region_name = os.environ.get('AWS_S3_REGION') or 'us-east-1'
        self.aws_s3_access_key = os.environ.get('AWS_S3_ACCESS_KEY_ID') \
            or aws_access_key_id
        self.aws_s3_secret_access_key = os.environ.get('AWS_S3_SECRET_ACCESS_KEY') or \
            aws_secret_access_key
        self.aws_s3_access_key_report_bucket = report_bucket or \
            os.environ.get('AWS_S3_ACCESS_KEY_ID_REPORT_BUCKET')
        self.aws_s3_secret_access_key_report_bucket = \
            os.environ.get('AWS_S3_SECRET_ACCESS_KEY_REPORT_BUCKET') or report_bucket
        self.aws_s3_access_key_npm_bucket = \
            os.environ.get('AWS_S3_ACCESS_KEY_ID_NPM_BUCKET')
        self.aws_s3_secret_access_key_npm_bucket = \
            os.environ.get('AWS_S3_SECRET_ACCESS_KEY_NPM_BUCKET')
        self.aws_s3_access_key_mvn_bucket = \
            os.environ.get('AWS_S3_ACCESS_KEY_ID_MVN_BUCKET')
        self.aws_s3_secret_access_key_mvn_bucket = \
            os.environ.get('AWS_S3_SECRET_ACCESS_KEY_MVN_BUCKET')
        self.aws_s3_access_key_pypi_bucket = \
            os.environ.get('AWS_S3_ACCESS_KEY_ID_PYPI_BUCKET')
        self.aws_s3_secret_access_key_pypi_bucket = \
            os.environ.get('AWS_S3_SECRET_ACCESS_KEY_PYPI_BUCKET')
        self.aws_s3_access_key_golang_bucket = \
            os.environ.get('AWS_S3_ACCESS_KEY_ID_GOLANG_BUCKET')
        self.aws_s3_secret_access_key_golang_bucket = \
            os.environ.get('AWS_S3_SECRET_ACCESS_KEY_GOLANG_BUCKET')
        self.deployment_prefix = os.environ.get('DEPLOYMENT_PREFIX') or 'dev'
        self.report_bucket_name = os.environ.get('REPORT_BUCKET_NAME')
        self.manifests_bucket = os.environ.get('MANIFESTS_BUCKET')
        if self.aws_s3_secret_access_key is None or self.aws_s3_access_key is None or\
                self.region_name is None or self.deployment_prefix is None:
            raise ValueError("AWS credentials or S3 configuration was "
                             "not provided correctly. Please set the AWS_S3_REGION, "
                             "AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY, REPORT_BUCKET_NAME "
                             "and DEPLOYMENT_PREFIX correctly.")
        # S3 endpoint URL is required only for local deployments
        self.s3_endpoint_url = os.environ.get('S3_ENDPOINT_URL') or 'http://localhost'

    def s3_client(self, bucket_name):
        """Provide s3 client for each bucket."""
        if bucket_name == os.environ.get('REPORT_BUCKET_NAME'):
            s3 = boto3.resource('s3', region_name=self.region_name,
                                aws_access_key_id=self.aws_s3_access_key_report_bucket,
                                aws_secret_access_key=self.aws_s3_secret_access_key_report_bucket)
        elif bucket_name == os.getenv('PYPI_MODEL_BUCKET'):
            s3 = boto3.resource('s3', region_name=self.region_name,
                                aws_access_key_id=self.aws_s3_access_key_pypi_bucket,
                                aws_secret_access_key=self.aws_s3_secret_access_key_pypi_bucket)
        elif bucket_name == os.getenv('GOLANG_MODEL_BUCKET'):
            s3 = boto3.resource('s3', region_name=self.region_name,
                                aws_access_key_id=self.aws_s3_access_key_golang_bucket,
                                aws_secret_access_key=self.aws_s3_secret_access_key_golang_bucket)
        elif bucket_name == os.getenv('MAVEN_MODEL_BUCKET'):
            s3 = boto3.resource('s3', region_name=self.region_name,
                                aws_access_key_id=self.aws_s3_access_key_mvn_bucket,
                                aws_secret_access_key=self.aws_s3_secret_access_key_mvn_bucket)
        elif bucket_name == os.getenv('NPM_MODEL_BUCKET'):
            s3 = boto3.resource('s3', region_name=self.region_name,
                                aws_access_key_id=self.aws_s3_access_key_npm_bucket,
                                aws_secret_access_key=self.aws_s3_secret_access_key_npm_bucket)
        else:
            s3 = boto3.resource('s3', region_name=self.region_name,
                                aws_access_key_id=self.aws_s3_access_key,
                                aws_secret_access_key=self.aws_s3_secret_access_key)
        return s3

    def store_json_content(self, content, bucket_name, obj_key):
        """Store the report content to the S3 storage."""
        s3 = self.s3_client(bucket_name)
        try:
            logger.info('Storing the report into the S3 file %s' % obj_key)
            s3.Object(bucket_name, obj_key).put(
                Body=json.dumps(content, indent=2).encode('utf-8'))
        except Exception as e:
            logger.exception('%r' % e)

    def read_json_object(self, bucket_name, obj_key):
        """Get the report json object found on the S3 bucket."""
        s3 = self.s3_client(bucket_name)
        try:
            obj = s3.Object(bucket_name, obj_key)
            result = json.loads(obj.get()['Body'].read().decode('utf-8'))
            return result
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.exception('No Such Key %s exists' % obj_key)
            elif e.response['Error']['Code'] == 'NoSuchBucket':
                logger.exception('ERROR - No Such Bucket %s exists' % bucket_name)
            else:
                logger.exception('%r' % e)
            return None

    def list_objects(self, bucket_name, frequency):
        """Fetch the list of objects found on the S3 bucket."""
        s3 = self.s3_client(bucket_name)
        prefix = '{dp}/{freq}'.format(dp=self.deployment_prefix, freq=frequency)
        res = {'objects': []}

        try:
            for obj in s3.Bucket(bucket_name).objects.filter(Prefix=prefix):
                if os.path.basename(obj.key) != '':
                    res['objects'].append(obj.key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.exception('ERROR - No Such Key %s exists' % prefix)
            elif e.response['Error']['Code'] == 'NoSuchBucket':
                logger.exception('ERROR - No Such Bucket %s exists' % bucket_name)
            else:
                logger.exception('%r' % e)

        return res
