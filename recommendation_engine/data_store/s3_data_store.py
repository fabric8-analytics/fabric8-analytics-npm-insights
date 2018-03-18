import json
import pickle
import os
import daiquiri
import boto3
import logging
import botocore
# import pandas as pd
import numpy as np
import zipfile
from recommendation_engine.utils import generic_utils as utils
from scipy.io import loadmat

daiquiri.setup(level=logging.ERROR)
_logger = daiquiri.getLogger(__name__)


class S3DataStore():
    def __init__(self, src_bucket_name, access_key, secret_key):
        """Create a new S3 data store instance.

        :src_bucket_name: The name of S3 bucket to connect to
        :access_key: The access key for S3
        :secret_key: The secret key for S3

        :returns: An instance of the S3 data store class
        """
        self.session = boto3.session.Session(aws_access_key_id=access_key,
                                             aws_secret_access_key=secret_key)
        self.s3_resource = self.session.resource('s3', config=botocore.client.Config(
            signature_version='s3v4'), region_name='us-east-1')
        self.bucket = self.s3_resource.Bucket(src_bucket_name)
        self.bucket_name = src_bucket_name

    def get_name(self):
        return "S3:" + self.bucket_name

    def read_json_file(self, filename):
        """Read JSON file from the S3 bucket"""
        return json.loads(self.read_generic_file(filename))

    def read_generic_file(self, filename):
        """Read a file from the S3 bucket."""
        obj = self.s3_resource.Object(self.bucket_name, filename).get()['Body'].read()
        utf_data = obj.decode("utf-8")
        return utf_data

    def list_files(self, prefix=None, max_count=None):
        """List all the files in the S3 bucket"""

        list_filenames = []
        if prefix is None:
            objects = self.bucket.objects.all()
            if max_count is None:
                list_filenames = [x.key for x in objects]
            else:
                counter = 0
                for obj in objects:
                    list_filenames.append(obj.key)
                    counter += 1
                    if counter == max_count:
                        break
        else:
            objects = self.bucket.objects.filter(Prefix=prefix)
            if max_count is None:
                list_filenames = [x.key for x in objects]
            else:
                counter = 0
                for obj in objects:
                    list_filenames.append(obj.key)
                    counter += 1
                    if counter == max_count:
                        break

        return list_filenames

    def read_all_json_files(self):
        """Read all the files from the S3 bucket"""
        list_filenames = self.list_files(prefix=None)
        list_contents = []
        for file_name in list_filenames:
            contents = self.read_json_file(filename=file_name)
            list_contents.append((file_name, contents))
        return list_contents

    def write_json_file(self, filename, contents):
        """Write JSON file into S3 bucket"""
        self.s3_resource.Object(self.bucket_name, filename).put(
            Body=json.dumps(contents))
        return None

    def write_pickle_file(self, complete_filename, pickle_filename):
        """Write Pickle file into S3 bucket"""

        self.s3_resource.Object(self.bucket_name, complete_filename).put(
            Body=open(os.path.join('/tmp', pickle_filename), 'rb'))

    def load_pickle_file(self, filename):
        """Load Pickle file from S3 bucket"""

        pickle_obj = self.s3_resource.Object(self.bucket_name, filename).get()[
            'Body'].read()
        return pickle.loads(pickle_obj)

    def upload_file(self, src, target):
        """Upload file into data store"""
        self.bucket.upload_file(src, target)
        return None

    def download_file(self, src, target):
        """Download file from data store"""
        self.bucket.download_file(src, target)
        return None

#     def write_pandas_df_into_json_file(self, data, filename):
        # self.write_json_file(filename=filename, contents=data.to_json())
        # return None

    # def read_json_file_into_pandas_df(self, filename):
        # json_string = self.read_json_file(filename=filename)
        # return pd.read_json(json_string, dtype=np.int8)

    def iterate_bucket_items(self, ecosystem='npm'):
        """
        Generator that iterates over all objects in a given s3 bucket

        See:
        https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2
        for return data format
        :param bucket: name of s3 bucket
        :return: dict of metadata for an object
        """
        client = self.session.client('s3')
        page = client.list_objects_v2(Bucket=self.bucket_name, Prefix=ecosystem)
        yield [obj['Key'] for obj in page['Contents']]
        while page['IsTruncated'] is True:
            page = client.list_objects_v2(Bucket=self.bucket_name, Prefix=ecosystem,
                                          ContinuationToken=page['NextContinuationToken'])
            yield [obj['Key'] for obj in page['Contents']]

    def list_folders(self, prefix=None):
        client = self.session.client('s3')
        result = client.list_objects(Bucket=self.bucket_name, Prefix=prefix + '/', Delimiter='/')
        folders = result.get('CommonPrefixes')
        if not folders:
            return []
        return [folder['Prefix'] for folder in folders]

    def upload_folder_to_s3(self, folder_path, zipfilename, prefix=''):
        """Zip a folder and upload it to s3.

        :folder_path: The local path of the folder to upload to s3
        """
        s3_client = self.session.client('s3')
        ziph = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED)
        utils.zipdir(folder_path, ziph)
        self.bucket.upload_file(zipfilename, os.path.join(prefix, zipfilename))

    def load_matlab_multi_matrix(self, s3_path):
        """This function loads a '.mat' & returns a dict representation.

        :s3_path: The path of the object in the S3 bucket.
        :returns: A dict containing numpy matrices against the kets of the
                  multi-matrix.
        """
        local_filename = os.path.join('/tmp', s3_path.split('/')[-1])
        self.download_file(s3_path, local_filename)
        model_dict = loadmat(local_filename)
        if not model_dict:
            _logger.error("Unable to load the model for scoring")
        return model_dict
