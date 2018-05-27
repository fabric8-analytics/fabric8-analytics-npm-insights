#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset object definition for the content representation data of CVAE.

Copyright © 2018 Red Hat Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import tensorflow as tf
import numpy as np

from recommendation_engine.utils.fileutils import load_sparse
import recommendation_engine.config.path_constants as path_constants


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        """Create a new hook."""
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        assert callable(self.iterator_initializer_func)
        self.iterator_initializer_func(session)


class PackageReconstructionInputFunction:
    """Define the train_input_function as a callable class."""

    def __init__(self, data, batch_size, num_epochs, mode, scope):
        """Create a new train_input_fn callable for our package tag dataset."""
        self.data = data
        self.batch_size = batch_size
        self.mode = mode
        self.scope = scope
        self.num_epochs = num_epochs
        self.init_hook = IteratorInitializerHook()

    def _build_dataset(self, tensor_tuple):
        """Build the dataset using tensors."""
        return tf.data.Dataset.from_tensor_slices(tensor_tuple)

    def __call__(self):
        """Return the next training example."""
        with tf.name_scope(self.scope):
            # Define placeholders
            placeholders = self._create_placeholders()

            # Build dataset iterator
            dataset = self._build_dataset(placeholders)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.shuffle(buffer_size=10000)
                dataset = dataset.repeat(self.num_epochs)
            dataset = dataset.batch(self.batch_size).prefetch(2)

            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()

            def _init(sess):
                print("Initializing the iterator.")
                sess.run(iterator.initializer,
                         feed_dict=self._get_feed_dict(placeholders))

            self.init_hook.iterator_initializer_func = _init

        return next_example, next_label

    def _create_placeholders(self):
        images_placeholder = tf.placeholder(self.data.dtype, self.data.shape,
                                            name='input_image')
        labels_placeholder = tf.placeholder(self.data.dtype, self.data.shape,
                                            name='reconstruct_image')
        return images_placeholder, labels_placeholder

    def _get_feed_dict(self, placeholders):
        assert len(placeholders) == 2
        return dict(zip(placeholders, [self.data, self.data]))


class CorruptedInputDecorator(PackageReconstructionInputFunction):
    """Corrupts input with noise, required for the pretraining DAE.

    Parameters
    ----------
    input_function : BaseInputFunction
        Input function to wrap.

    noise_factor : float
        Amount of noise to apply.

    """

    def __init__(self, input_function, noise_factor=0.5):
        """Create a new input corruptor."""
        super().__init__(data=input_function.data,
                         batch_size=input_function.batch_size,
                         num_epochs=input_function.num_epochs,
                         mode=input_function.mode,
                         scope=input_function.scope)
        self.input_function = input_function
        self.noise_factor = noise_factor

    def _build_dataset(self, tensor_tuple):
        dataset = self.input_function._build_dataset(tensor_tuple)

        def add_noise(input_rep, output_rep):
            noise = self.noise_factor * tf.random_normal(input_rep.shape.as_list())
            input_corrupted = tf.clip_by_value(tf.add(input_rep, noise), 0., 1.)
            return input_corrupted, output_rep

        # run mapping function in parallel
        dataset = dataset.map(add_noise, num_parallel_calls=4)
        return dataset


class PackageTagRepresentationDataset:
    """Package representation data set for learning an unsupervised autoencoder."""

    @staticmethod
    def _input_fn_corrupt(data, batch_size, num_epochs, mode, scope, noise_factor=0.0):
        f = PackageReconstructionInputFunction(data, batch_size, num_epochs,
                                               mode, scope)
        if noise_factor > 0:
            return CorruptedInputDecorator(f, noise_factor=noise_factor)
        return f

    @classmethod
    def get_train_input_fn(cls, batch_size, num_epochs, mode, scope, data_store, noise_factor=0.0):
        """Return the train_input_function required by the estimator."""
        # TODO: Switch to sparse_tensor once there is more information around the dataset API.
        data = cls._convert_sparse_matrix_to_array(load_sparse(
            path_constants.DATA_SPARSE_REP, data_store))
        return cls._input_fn_corrupt(data, batch_size, num_epochs, mode, scope, noise_factor)

    @staticmethod
    def _convert_sparse_matrix_to_sparse_tensor(sparse_mat_scipy):
        """
        Convert a sparse matrix to a sparse tensor.

        :returns: A sparse tensor representation of a sparse csr_matrix.
        :rtype: tf.SparseTensor
        """
        coo = sparse_mat_scipy.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        print("Initializing a sparse tensor with shape: {}".format(coo.shape))
        return tf.SparseTensor(indices, coo.data, coo.shape)

    @staticmethod
    def _convert_sparse_matrix_to_array(sparse_mat_scipy):
        """
        Convert a sparse matrix to a numpy ndarray.

        :returns: A sparse tensor representation of a sparse csr_matrix.
        :rtype: tf.SparseTensor
        """
        return sparse_mat_scipy.toarray()
