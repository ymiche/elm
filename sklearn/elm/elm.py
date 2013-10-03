# -*- coding: utf8
"""Extreme Learning Machines

Some description of the package here.

"""
# Authors: Yoan Miche <yoan.miche@aalto.fi>
# License: BSD 3 clause

from __future__ import division
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from sklearn.utils import as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from ..utils.validation import check_arrays

from . import helpers

__all__ = ["elm"]



class elm(BaseEstimator, RegressorMixin): 
    """Extreme Learning Machine for classification and
    regression
    
    Assumes the data is normalized already (input x), i.e. zero mean and unit
    variance.

    Some more detail here...

    Parameters
    ----------
    problem : string, either 'r' for regression, or 'c' for classification,
              defaults to regression 'r'
              Tells if the problem is regression (default) or classification.

    n_neurons : int, strictly positive, defaults to 100
                Number of neurons to be used by the model (default: 100)

    activation_function : function handle, should respect ELM hypotheses (see
    ref)
                          Activation function for the neurons (default: np.tanh)

    Notes
    -----
    This version automatically adds linear neurons to the ELM.
    
    Attributes
    ----------
    `inputWeights` : 
    `inputBiases` :
    `hiddenLayer` :
    `outputWeights` :
    `trained` :

    Examples
    --------

    See also
    --------

    References
    ----------


    """

    def __init__(self, problem='r', n_neurons=100, activation_function=np.tanh):
        if not problem in ['r','c']:
            raise ValueError('Unknown type of problem. See help.')
        self.problem = problem
        if not isinstance(n_neurons, int):
            raise ValueError('Given number of neurons is not an integer. See \
                    help.')
        if not n_neurons > 0:
            raise ValueError('Given number of neurons should be strictly \
                    positive. See help.')
        self.n_neurons=n_neurons
        if not isinstance(activation_function, np.ufunc):
            raise ValueError('Given activation function is not a function. See \
                    help.')
        self.activation_function = activation_function


    def fit(self, x, y):
        """Fit the ELM model according to the given training data.
            
        Parameters
        ----------
        x : array-like, dense, shape = [n_samples, n_features]
            The training vectors for the ELM model, with n_samples the number
            of samples and n_features the number of features
            (attributes/dimensions)
        
        y : array-like, dense, shape = [n_samples, n_outputs]
            The output, target values, for each of the samples. The output can
            be multi-dimensional. The case where y has only one dimension
            (y.shape = (n_samples)) will be converted to y.shape = (n_samples,
            1) for the algorithm.
        
        Returns
        -------
        self : object
               Returns self

        yh : array-like, shape = [n_samples, n_outputs]
             The training output of the model
        
        yloo : array-like, shape = [n_samples, n_outputs]
               The Leave-One-Out estimation of the output by the model

        Notes
        -----

        """

        # Check the input data for inconsistency
        x, y = check_arrays(x, y, sparse_format='dense')
        # Get the number of samples and dimensionality of the data
        self.n_samples, self.n_features = x.shape
        # Check the structure of the given output y and convert to a 2-D array
        _yWasOneDimensional = False
        if len(y.shape) == 1:
            _yWasOneDimensional = True
            y = y.reshape((self.n_samples, 1))

        self.n_outputs = y.shape[1]


        # Generate random input weights
        self.inputWeights = np.random.randn(self.n_neurons, x.shape[1])*np.sqrt(3)
        # Generate random input biases
        self.inputBiases = np.random.randn(self.n_neurons, 1)*0.5
        # Project the input data
        self.hiddenLayer = np.dot(x, self.inputWeights.T)
        # Add the biases
        self.hiddenLayer += np.tile(self.inputBiases, (1, self.hiddenLayer.shape[0])).T
        # Non-linearity in the neurons (tanh function)
        self.hiddenLayer = self.activation_function(self.hiddenLayer)
        # Add linear neurons to the total
        self.hiddenLayer = np.hstack((self.hiddenLayer, x))
    
        # Check the hidden layer condition number before running this
        if np.linalg.cond(x) > 10**14:
            # TODO Give a better message for this, and point to methods that do
            # help
            raise ValueError('Hidden Layer is ill-conditioned. Consider \
                    reducing correlations in the input data by feature \
                    reduction/selection.')
        
        if self.problem == 'c':
            _y = y.copy()
            y, mapping = helpers.oneInAllMapping(y)
            # Setting them to -1 and 1 instead of 0 and 1
            y = y*2-1

        betas, residues, rank, s = np.linalg.lstsq(self.hiddenLayer, y)
        yh = np.dot(self.hiddenLayer, betas)
        yloo, errloo = helpers.leaveOneOut(self.hiddenLayer, y, betas)
        
        # Set the output weights 
        self.outputWeights = betas
        self.trained = True
       
        if self.problem == 'c':
            indices = np.argmax(yh, axis=1)
            yh = np.zeros(yh.shape)
            for i, index in enumerate(indices):
                yh[i, index] = 1
            yh = helpers.oneInAllDemapping(yh, mapping)

            indices = np.argmax(yloo, axis=1)
            yloo = np.zeros(yloo.shape)
            for i, index in enumerate(indices):
                yloo[i, index] = 1
            yloo = helpers.oneInAllDemapping(yloo, mapping)

        if _yWasOneDimensional:
            # Case where the given output was 1-D, return in the same format
            yh = yh.ravel()
            yloo = yloo.ravel()

        return self, yh, yloo
    

    def predict(self, x):
        """This function does prediction on an array of test vectors x.

        The predicted value for each sample in X is returned.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]

        Returns
        -------
        c : array, shape = [n_samples]
        """
        
        return y_pred


