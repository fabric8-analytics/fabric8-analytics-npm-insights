"""Tests for the class S3DataStore."""

import pytest

from recommendation_engine.data_store.s3_data_store import S3DataStore
from tests.s3_mocks import MockedS3Resource


def test_initial_state():
    """Check the initial state of S3DataStore object."""
    s3DataStore = S3DataStore("bucket", "access_key", "secret_key")
    assert s3DataStore


def test_get_name():
    """Check the method get_name()."""
    s3DataStore = S3DataStore("bucket", "access_key", "secret_key")
    assert s3DataStore.get_name() == "S3:bucket"


def test_read_json_file_positive():
    """Check the method read_json_file()."""
    s3DataStore = S3DataStore("bucket", "access_key", "secret_key")
    s3DataStore.s3_resource = MockedS3Resource()
    s3DataStore.read_json_file("file")


def test_read_json_file_negative():
    """Check the method read_json_file()."""
    # try for improper access_key and secret_key
    s3DataStore = S3DataStore("bucket", "access_key", "secret_key")
    with pytest.raises(Exception):
        s3DataStore.read_json_file("file")
