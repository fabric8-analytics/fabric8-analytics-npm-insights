#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset object definition for the content representation data of CVAE
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
from abc import abstractmethod, ABCMeta
from scipy import sparse

import tensorflow as tf
import numpy as np


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        assert callable(self.iterator_initializer_func)
        self.iterator_initializer_func(session)


class BaseInputFunction(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, data, batch_size, num_epochs, mode, scope):
        self.data = data
        self.batch_size = batch_size
        self.mode = mode
        self.scope = scope
        self.num_epochs = num_epochs
        self.init_hook = IteratorInitializerHook()

    @abstractmethod
    def _create_placeholders(self):
        """Returns placeholders for input data

        Returns
        -------
        data : tf.placeholder
            Placeholder for training data

        labels : tf.placeholder
            Placeholder for labels
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_feed_dict(self, placeholders):
        """Return feed_dict to initialize placeholders.

        Parameters
        ----------
        placeholders : list of tf.placeholder
            Placeholders to initialize

        Returns
        -------
        feed_dict : dict
            Dictionary with values used to initialize
            passed tf.placeholders.
        """
        raise NotImplementedError()

    def _build_dataset(self, placeholders):
        return tf.data.Dataset.from_tensor_slices(placeholders)

    def __call__(self):
        """This is the implementation for the train_input_function."""
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
                sess.run(iterator.initializer,
                         feed_dict=self._get_feed_dict(placeholders))

            self.init_hook.iterator_initializer_func = _init

        return next_example, next_label


class CorruptedInputDecorator(BaseInputFunction):
    """Corrupts input with noise

    Parameters
    ----------
    input_function : BaseInputFunction
        Input function to wrap.

    noise_factor : float
        Amount of noise to apply.
    """

    def __init__(self, input_function, noise_factor=0.5):
        super().__init__(data=input_function.data,
                         batch_size=input_function.batch_size,
                         num_epochs=input_function.num_epochs,
                         mode=input_function.mode,
                         scope=input_function.scope)
        self.input_function = input_function
        self.noise_factor = noise_factor

    def _create_placeholders(self):
        return self.input_function._create_placeholders()

    def _get_feed_dict(self, placeholders):
        return self.input_function._get_feed_dict(placeholders)

    def _build_dataset(self, placeholders):
        dataset = self.input_function._build_dataset(placeholders)

        def add_noise(input_img, groundtruth_img):
            noise = self.noise_factor * tf.random_normal(input_img.shape.as_list())
            input_corrupted = tf.clip_by_value(tf.add(input_img, noise), 0., 1.)
            return input_corrupted, groundtruth_img

        # run mapping function in parallel
        return dataset.map(add_noise, num_parallel_calls=4)


class MNISTReconstructionInputFunction(BaseInputFunction):
    """MNIST input function to train an autoencoder.

    Parameters
    ----------
    data : tensorflow.examples.tutorials.mnist.input_data
        MNIST dataset.

    mode : int
        Train, eval or prediction mode.

    scope : str
        Name of input function in Tensor board.
    """

    def __init__(self, data, batch_size, num_epochs, mode, scope):
        super().__init__(data=data, batch_size=batch_size,
                         num_epochs=num_epochs, mode=mode, scope=scope)
        self._images_placeholder = None
        self._labels_placeholder = None

    def _create_placeholders(self):
        images_placeholder = tf.placeholder(self.data.dtype, self.data.shape,
                                            name='input_image')
        labels_placeholder = tf.placeholder(self.data.dtype, self.data.shape,
                                            name='reconstruct_image')
        return images_placeholder, labels_placeholder

    def _get_feed_dict(self, placeholders):
        assert len(placeholders) == 2
        return dict(zip(placeholders, [self.data, self.data]))


class PackageTagRepresentationDataset:
    """MNIST data set for learning an unsupervised autoencoder.

    Parameters
    ----------
    data_dir : str
        Path to directory to write data to.

    noise_factor : float
        The amount of noise to apply. If non-zero, a denoising
        autoencoder will be trained.
    """
    @staticmethod
    def _input_fn_corrupt(cls, data, batch_size, num_epochs, mode, scope, noise_factor=0):
        f = MNISTReconstructionInputFunction(data, batch_size, num_epochs,
                                             mode, scope)
        if noise_factor > 0:
            return CorruptedInputDecorator(f, noise_factor=noise_factor)
        return f

    @classmethod
    def get_train_input_fn(cls, batch_size, num_epochs):
        """Return the train_input_function required by the estimator."""
        return cls._input_fn_corrupt(
           cls._convert_sc
           batch_size, num_epochs,
           tf.estimator.ModeKeys.TRAIN,
           'training_data')

    @staticmethod
    def _convert_scipy_sparse_to_dense(matrix_filename):
        """Convert a row of a scipy sparse matrix to its dense array representation."""
        package_tag_representation_sparse = sparse.load_npz(matrix_filename)
        _convert_sparse_matrix_to_sparse_tensor(package_tag_representation_sparse)

    @staticmethod
    def _convert_sparse_matrix_to_sparse_tensor(X):
        """
        :returns: A sparse tensor representation of a sparse csr_matrix.
        :rtype: tf.SparseTensor
        """
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
