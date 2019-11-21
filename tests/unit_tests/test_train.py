import numpy as np
import tensorflow as tf
from recommendation_engine.autoencoder.train import train
import boto3
# from moto import mock_s3
from unittest import mock

# class TestTrainNetwork(tf.test.TestCase):
test_data = np.load('tests/test_data/2019-01-03/trained-model/content_matrix.npy')

# @mock_s3
@mock.patch('train.TrainNetwork().get_preprocess_data', return_value=None)
def test_train(self, data=test_data):
    test_train_obj = train.TrainNetwork()
    # test_train_obj.get_preprocess_data=None
    test_train_obj.train(data)