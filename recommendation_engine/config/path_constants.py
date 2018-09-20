"""Contains the path constants for S3 and local storage."""
import os

ECOSYSTEM = os.environ.get('CHESTER_SCORING_REGION', 'npm')
PMF_MODEL_PATH = os.path.join(ECOSYSTEM, 'models/cvae-model/pmf-packagedata.mat')
PACKAGE_TO_ID_MAP = os.path.join(ECOSYSTEM, 'node-pmf-scoring/package_to_index_map.json')
ID_TO_PACKAGE_MAP = os.path.join(ECOSYSTEM, 'node-pmf-scoring/index_to_package_map.json')
PRECOMPUTED_STACKS = os.path.join(ECOSYSTEM, 'training-data-node/manifest_user_data.dat')
TRAINING_DATA_ITEMS = os.path.join(ECOSYSTEM, 'training-data-node/packagedata-train-5-items.dat')
PACKAGE_TAG_MAP = os.path.join(ECOSYSTEM, 'node-pmf-scoring/package_tag_map.json')
CVAE_MODEL_PATH = os.path.join(ECOSYSTEM, 'models/cvae-model/')
