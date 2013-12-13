#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Yoan Miche <yoan.miche@aalto.fi>
# Licence: BSD 3 clause

"""
The small functions required by the elm module. 
"""

import numpy as np

def leaveOneOut(x, y, w):
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
    if len(x.shape) != 2:
        raise ValueError('The shape of the input x must be 2-D. See help.')
    n_samples, n_features = x.shape
    if len(y.shape) > 2:
        raise ValueError('The shape of the output y must be 1 or 2-D. See help.')
    if len(y.shape) == 1:
        n_outputs = 1
    else:
        n_outputs = y.shape[1]

    C = np.dot(x.T, x)
    # Using the pseudo-inverse of numpy, as it is more stable than the direct
    # inverse (SVD based)
    P = np.dot(x, np.linalg.pinv(C))
    proxyDiagMatrix = np.diag(np.dot(P, x.T))
    S = y-np.dot(x, w)
    S = S/(1 - np.tile(proxyDiagMatrix.reshape((n_samples, 1)), n_outputs))
    errloo = np.mean(np.mean(np.power(S, 2), axis=0))
    yloo = y - S;
   
    return yloo, errloo

 
def oneInAllMapping(y):
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

    Note
    ----
    For the special case of only two classes, this returns not a one in all
    mapping, but a single column mapping, with 1 for the first class, and 0 for
    the other. The demapping handles this correctly as well.
    """
    if len(y.shape) == 1:
        raise ValueError('The shape of the output vector must be 2-D.')
    if y.shape[1] != 1:
        raise ValueError('Classification output vector cannot be \
                multi-output.')
    
    # Extract all the possible unique classes from the given vector 
    classes, indices = np.unique(y, return_inverse=True)
   
    if len(classes) == 2:
        b = np.zeros((y.shape[0], 1))
        mapping = np.zeros((len(classes), 2))
        b[indices==0, 0] = 1
        mapping[0, 1] = 1
        mapping[0, 0] = classes[0]
        mapping[1, 0] = classes[1]
    else:
        b = np.zeros((y.shape[0], len(classes)))
        mapping = np.zeros((len(classes), len(classes)+1))

        for i, theClass in enumerate(classes):
            # Code 1 in zeros for this one
            b[indices==i, i] = 1
            # Update the corresponding mapping
            mapping[i, i+1] = 1
            mapping[i, 0] = theClass

    return b, mapping


def oneInAllDemapping(b, mapping):
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

    if len(classes) == 2:
        y[b[:, 0] == 1, 0] = mapping[0, 0]
        y[b[:, 0] == 0, 0] = mapping[1, 0]
    else:
        for i, theClass in enumerate(classes):
            y[b[:, i] == 1, 0] = theClass

    return y

 
