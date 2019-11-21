import unittest
import numpy as np
import tensorflow as tf
from recommendation_engine.autoencoder.pretrain import pretrain


test_data = np.load('tests/test_data/2019-01-03/trained-model/content_matrix.npy')
def test_fit(data = test_data):
    pretrain.fit(data)

