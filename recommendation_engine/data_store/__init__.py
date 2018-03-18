import recommendation_engine.config.cloud_constants as cloud_constants
from recommendation_engine.data_store.s3_data_store import S3DataStore
from recommendation_engine.data_store.local_filesystem import LocalFileSystem


if cloud_constants.USE_CLOUD_SERVICES:
    data_store_wrapper = S3DataStore(src_bucket_name=cloud_constants.S3_BUCKET_NAME,
                                     access_key=cloud_constants.AWS_S3_ACCESS_KEY_ID,
                                     secret_key=cloud_constants.AWS_S3_SECRET_KEY_ID)
else:
    data_store_wrapper = LocalFileSystem()
