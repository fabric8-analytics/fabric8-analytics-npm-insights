"""Tests for the class S3DataStore."""

from recommendation_engine.data_store.s3_data_store import S3DataStore


def test_initial_state():
    """Check the initial state of S3DataStore object."""
    s3DataStore = S3DataStore("bucket", "access_key", "secret_key")
    assert s3DataStore
