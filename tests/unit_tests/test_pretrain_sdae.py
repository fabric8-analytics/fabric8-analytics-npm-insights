import numpy as np
import tensorflow as tf
from recommendation_engine.autoencoder.pretrain import pretrain_sdae
from recommendation_engine.autoencoder.pretrain import pretrain_latent


class TestPretrainNetwork(tf.test.TestCase):

    test_data = np.load('tests/test_data/2019-01-03/trained-model/content_matrix.npy')

    def test_train(self, data=test_data):
        test_latent = pretrain_latent.PretrainLatent()
        test_latent.train(data)
        test_latent_weights, test_de_latent = test_latent.get_weights()
        test_pretrain_latent = test_latent_weights+test_de_latent[::-1]
        test_sdae_obj = pretrain_sdae.PretrainNetwork(hidden_dim=[200, 100], pretrain=test_pretrain_latent)
        test_sdae_obj.train(data)

