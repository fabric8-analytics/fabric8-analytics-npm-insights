
import numpy as np
from recommendation_engine.autoencoder.pretrain import pretrain_hidden
import tensorflow as tf
import recommendation_engine.config.params_training as train_params
from recommendation_engine.config.path_constants import TEMPORARY_MODEL_PATH
data = np.load('tests/test_data/2019-01-03/trained-model/content_matrix.npy')


class TestPretrainHidden(tf.test.TestCase):

    data = np.load('tests/test_data/2019-01-03/trained-model/content_matrix.npy')
    test_shape = np.shape(data)[1]

    def test_train(self, test_data=data, test_shape=test_shape):
        test_hidden_obj = pretrain_hidden.PretrainHidden(layer_dim=10)
        test_out = test_hidden_obj.train(data=test_data, input_dimension=test_shape)
        self.assertEqual(np.shape(test_out)[0], 135)
        self.assertEqual(test_hidden_obj.layer_dim, 10)

