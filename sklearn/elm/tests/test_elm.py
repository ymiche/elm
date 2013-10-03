"""
Testing for ELM module (sklearn.elm)
"""

# Author: Yoan Miche <yoan.miche@aalto.fi> 
# Licence: BSD 3 clause

from nose.tools import raises
from nose.tools import assert_raises
from nose.tools import assert_true
from nose.tools import assert_less

import numpy as np

from sklearn.elm import elm
from sklearn import datasets

def test_init_wrong_input():
    """test_init_wrong_input : Test the input parameters for the class constructor
    """
    # Test the parameters and the input types
    # Wrong type
    assert_raises(ValueError, elm, problem=2)
    # Wrong entry
    assert_raises(ValueError, elm, problem='wrong')
    # Wrong type
    assert_raises(ValueError, elm, n_neurons='number')
    # Wrong entry
    assert_raises(ValueError, elm, n_neurons=-2)
    # Wrong type
    assert_raises(ValueError, elm, activation_function='function')
    # Wrong entry
    assert_raises(ValueError, elm, activation_function=np.max)


def test_data_shapes():
    """test_data_shapes : Test for different types of input data
    """
    # Test the regression part of the code, first
    n_neurons = 100
    problem = 'r'
    activation_function = np.tanh
    myelm = elm(problem, n_neurons, activation_function)
    # Generate 1-d regression data, first
    x = np.random.randn(200, 1)
    y = np.random.randn(200, 1)
    myelm.fit(x, y)
    # Then n-d regression data
    x = np.random.randn(200, 5)
    y = np.random.randn(200, 1)
    myelm.fit(x, y)
    # And n-d regression data with multi-output
    x = np.random.randn(200, 5)
    y = np.random.randn(200, 3)
    myelm.fit(x, y)

    # Now classification data
    problem = 'c'
    myelmC = elm(problem, n_neurons, activation_function)
    # 1-d binary classification
    iris = datasets.load_iris()
    xall = iris['data']
    x = xall[:, 0].reshape((xall.shape[0], 1))
    yall = iris['target']
    y = np.sign(np.random.randn(x.shape[0], 1))
    myelmC.fit(x, y)
    # n-d binary classification
    x = xall
    # n-d multi-class classification
    # n-d multi-class multi-output classification


def test_fit_1d():
    """test_fit_1d : Test the fit of the model on a defined example, with known parameters.
    """
    # Generate the data
    x = np.random.rand(1000,1)
    y = np.sin(4*x+3)*np.sin(15*x+9)
    n_neurons = 100
    problem = 'r'
    activation_function = np.tanh
    myelm = elm(problem, n_neurons, activation_function)
    # The fit should always be better than 1e-8, in MSE
    myelm, yh, yloo = myelm.fit(x, y)
    assert_less(np.mean((yh-y)**2),10**(-8))
    assert_less(np.mean((yloo-y)**2), 10**(-8))
