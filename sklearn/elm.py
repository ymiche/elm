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

from .base import BaseEstimator, TransformerMixin
from .externals import six
from .externals.six.moves import xrange
from .utils import check_random_state
from .utils.extmath import safe_sparse_dot
from .utils.random import sample_without_replacement
from .utils.validation import check_arrays


__all__ = ["ExtremeLearningMachine"]

def myLeaveOneOut(X, Y):
    # Verify inputs
    # TODO Replace the following by calls to the validation package
    if not isinstance(X, np.ndarray):
        raise Exception('Given array is not a Numpy Array.')
    if not isinstance(Y, np.ndarray):
        raise Exception('Given array is not a Numpy Array.')
    if not len(X.shape) == 2:
        raise Exception('Given X input data is not a 2D matrix.')
    if not len(Y.shape) == 2:
        raise Exception('Given Y output data is not a 2D matrix.')
    if not X.shape[0] == Y.shape[0]:
        raise Exception('Number of samples of input X and output Y do not match.')
    if not Y.shape[1] == 1:
        raise Exception('Multi-dimensional output not supported in this function.')

    # Directly translated from the matlab function of the same name.
    # Optimizations likely possible :)
    N, d = X.shape
    C = np.dot(X.T, X)
    XCondNumber = 1/np.linalg.cond(X)

    # If the input matrix is normally conditioned
    if XCondNumber > 10^(-14):
        W, residues, rank, s = np.linalg.lstsq(X, Y)
        P = np.dot(X, np.linalg.inv(C))
        proxyDiagMatrix = np.diag(np.dot(P, X.T))
        S = Y-np.dot(X, W)
        S = S/(1-proxyDiagMatrix)
        ErrLOO = np.mean(np.power(S, 2))
        YLOO = Y - S;
    # Otherwise, we just don't perform the LOO
    else:
        W = np.zeros((d, 1))
        YLOO = np.zeros((N, 1))
        ErrLOO = np.inf

    return W, YLOO, ErrLOO



class ExtremeLearningMachine(object):
    """Extreme Learning Machine for classification and regression

    Some more detail here...

    Parameters
    ----------


    Examples
    --------

    See also
    --------
    """

    def __init__(self, n_neurons=100, activation_function=np.tanh):
        self.n_neurons=n_neurons
        self.activation_function = activation_function

    def fit(self, X, Y):
        # Actual training using available training data
        # Generate random input weights
        self.inputWeights = np.random.randn(self.n_neurons, X.shape[1])*np.sqrt(3)
        # Generate random input biases
        self.inputBiases = np.random.randn(self.n_neurons, 1)*0.5
        # Project the input data
        self.hiddenLayer = np.dot(X, self.inputWeights.T)
        # Add the biases
        self.hiddenLayer += np.tile(self.inputBiases, (1, self.hiddenLayer.shape[0])).T
        # Non-linearity in the neurons (tanh function)
        self.hiddenLayer = self.activation_function(self.hiddenLayer)
        # Add linear neurons to the total
        self.hiddenLayer = np.hstack((self.hiddenLayer, X))

        betas, yloo, dump2 = myLeaveOneOut(self.hiddenLayer, Y)


        # Set the output weights 
        self.outputWeights = betas

        self.trained = True
        return yloo

