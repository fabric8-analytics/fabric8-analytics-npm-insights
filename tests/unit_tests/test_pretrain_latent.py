import unittest
import numpy as np
import tensorflow as tf
from recommendation_engine.autoencoder.pretrain import pretrain_latent


class TestPretrainLatent(tf.test.TestCase):

    test_data = np.load('tests/test_data/2019-01-03/trained-model/content_matrix.npy')

    def test_train(self, data=test_data):
        self.latent_obj = pretrain_latent.PretrainLatent()
        self.latent_obj.train(data)
        self.assertEqual(np.shape(self.latent_obj.weights), (2, 2))
        self.assertEqual(np.shape(self.latent_obj.de_weights), (1, 2))

