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
        self.problem = problem
        self.n_neurons=n_neurons
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
            y = y.reshape((n_samples, 1))

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
            y, mapping = self._oneInAllMapping(y)
            # Setting them to -1 and 1 instead of 0 and 1
            y = y*2-1

        betas, residues, rank, s = np.linalg.lstsq(self.hiddenLayer, y)
        yh = np.dot(self.hiddenLayer, betas)
        yloo, errloo = self._leave_one_out(self.hiddenLayer, y, betas)
        
        # Set the output weights 
        self.outputWeights = betas
        self.trained = True
       
        if self.problem == 'c':
            indices = np.argmax(yh, axis=1)
            yh = np.zeros(yh.shape)
            for i, index in enumerate(indices):
                yh[i, index] = 1
            yh = self._oneInAllDemapping(yh, mapping)

            indices = np.argmax(yloo, axis=1)
            yloo = np.zeros(yloo.shape)
            for i, index in enumerate(indices):
                yloo[i, index] = 1
            yloo = self._oneInAllDemapping(yloo, mapping)

        if _yWasOneDimensional:
            # Case where the given output was 1-D, return in the same format
            yh = yh.ravel()
            yloo = yloo.ravel()

        return self, yh, yloo
    

    def _leave_one_out(x, y, w):
        """Performs the PRESS Leave One Out calculation, from D. Allen's formula.
        This algorithm evaluates the LOO output of a linear system (exact
        value, not an estimation). This only works for a linear system between
        x and y.

        The algorithm expects the system to be properly conditioned, i.e.
        numpy.linalg.cond(x) < 10^14 (personal arbitrary limit).

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            The input of the linear system for which to calculate the LOO
            output

        y : array-like, shape = [n_samples, n_outputs]
            The output of the linear system to solve

        w : array-like, shape = [n_features, n_outputs]
            The solution of the system xw=y, computed by numpy.linalg.lstsq
            typically

        Returns
        -------
        yloo : array-like, shape = [n_samples, n_outputs]
               The Leave-One-Out output of the system
        
        errloo : double
                 The Mean Square Error between the real output and the LOO one

        Reference
        ---------
        David M. Allen. The relationship between variable selection and data
        augmentation and a method for prediction. Technometrics, 16(1):125–127,
        February 1974.
        """
        # Optimizations likely possible :)
        C = np.dot(x.T, x)
        P = np.dot(x, np.linalg.inv(C))
        proxyDiagMatrix = np.diag(np.dot(P, x.T))
        S = y-np.dot(x, w)
        S = S/(1 - proxyDiagMatrix)
        errloo = np.mean(np.power(S, 2))
        yloo = y - S;
       
        return yloo, errloo

    
    def _oneInAllMapping(y):
        """Create a One in All code for classification vector.
        Only works for single-output multi-class classification.

        Parameters
        ----------
        y : array-like, shape = [n_samples, 1]
            The vector of classes to encode

        Returns
        -------
        b : array-like, shape = [n_samples, <number of classes in y>]
            The matrix of the mapped values to a one in all code

        mapping : array-like, shape = [<number of classes>, <number of classes in y>+1]
                  The matrix used to map the values of the classes to the codes


        """
        if len(y.shape) == 1:
            raise ValueError('The shape of the output vector must be 2-D.')
        if y.shape[1] != 1:
            raise ValueError('Classification output vector cannot be \
                    multi-output.')
        
        # Extract all the possible unique classes from the given vector 
        classes, indices = np.unique(y, return_inverse=True)
        
        b = np.zeros((y.shape[0], len(classes)))
        mapping = np.zeros((len(classes), len(classes)+1))

        for i, theClass in enumerate(classes):
            # Code 1 in zeros for this one
            b[indices==i, i] = 1
            # Update the corresponding mapping
            mapping[i, i+1] = 1
            mapping[i, 0] = theClass

        return b, mapping

    def _oneInAllDemapping(b, mapping):
        """Demaps the representation done by the oneInAllMapping function
        using the provided mapping.

        Parameters
        ----------
        b : array-like, shape = [n_samples, <number of classes>]
            The matrix of the mapped values, this is a binary matrix

        mapping: array-like, shape = [<number of classes>, <number of classes>+1]
                 The matrix used to map the values of the classes to the codes

        Returns
        -------
        y : array-like, shape = [n_samples, 1]
            The decoded vector of classes
        """
        classes = mapping[:,0]

        y = np.zeros((b.shape[0], 1))

        for i, theClass in enumerate(classes):
            y[b[:, i] == 1, 0] = theClass

        return y

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
