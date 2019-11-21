"""Contains the path constants for S3 and local storage."""
import os

MODEL_VERSION = os.environ.get('MODEL_VERSION', '2019-01-03')
TEMPORARY_PATH = '/tmp/trained-model/'
TEMPORARY_MODEL_PATH = '/tmp/intermediate-model/'
TEMPORARY_LATENT_PATH = os.path.join(TEMPORARY_MODEL_PATH, 'latent_pretrain_all')
TEMPORARY_SDAE_PATH = os.path.join(TEMPORARY_MODEL_PATH, 'train')
TEMPORARY_CVAE_PATH = os.path.join(TEMPORARY_MODEL_PATH, 'cvae')
TEMPORARY_PMF_PATH = os.path.join(TEMPORARY_MODEL_PATH, 'pmf-packagedata.mat')
USER_ITEM_FILEPATH = os.path.join(TEMPORARY_PATH, "user_train_data.dat")
ITEM_USER_FILEPATH = os.path.join(TEMPORARY_PATH, "item_train_data.dat")
PRECOMPUTED_MANIFEST_PATH = os.path.join(TEMPORARY_PATH, "manifest_user_data.dat")
PMF_MODEL_PATH = os.path.join(MODEL_VERSION, 'intermediate-model/pmf-packagedata.mat')
PACKAGE_TO_ID_MAP = os.path.join(MODEL_VERSION, 'trained-model/package_to_index_map.json')
ID_TO_PACKAGE_MAP = os.path.join(MODEL_VERSION, 'trained-model/index_to_package_map.json')
PACKAGE_TAG_MAP = os.path.join(MODEL_VERSION, 'trained-model/package_tag_map.json')
