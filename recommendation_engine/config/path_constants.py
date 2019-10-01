"""Contains the path constants for S3 and local storage."""
import os

ECOSYSTEM = os.environ.get('CHESTER_SCORING_REGION', 'npm')
MODEL_VERSION = os.environ.get('MODEL_VERSION', '2019-01-03')
DEPLOYMENT_PREFIX = os.environ.get('DEPLOYMENT_PREFIX', 'dev')
TEMPORARY_PATH = '/tmp/trained-model'
PMF_MODEL_PATH = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                              'intermediate-model/cvae-model/pmf-packagedata.mat')
PACKAGE_TO_ID_MAP = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                                 'trained-model/package_to_index_map.json')
ID_TO_PACKAGE_MAP = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                                 'trained-model/index_to_package_map.json')
PRECOMPUTED_STACKS = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                                  'data/manifest_user_data.dat')
TRAINING_DATA_ITEMS = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                                   'data/packagedata-train-5-items.dat')
PACKAGE_TAG_MAP = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                               'trained-model/package_tag_map.json')
CVAE_MODEL_PATH = os.path.join(ECOSYSTEM, DEPLOYMENT_PREFIX, MODEL_VERSION,
                               'intermediate-model/cvae-model/')
