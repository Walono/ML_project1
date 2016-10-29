# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def add_feature(y, x, tX, method,  tX_test, x_test, **kwargs):
    """ function that adds method(x) to tX if corr(y, method(x)) > corr(y, prev_method(x))) """
    prev_corr = abs(np.corrcoef(y,x)[1,0])
    
    x_transform = method(x, **kwargs)
    x_transform_test = method(x_test, **kwargs)
    
    new_corr = abs(np.corrcoef(y,x_transform[0])[1,0])

    if new_corr > prev_corr :
        tX = np.vstack((tX, x_transform[0]))
        tX_test = np.vstack((tX_test, x_transform_test[0]))
        tX_test = np.nan_to_num(tX_test)

    return tX, tX_test

def sqrt_def(x, **kwargs):
    """ Apply sqrt on x """
    new = np.array([np.sqrt(x)])
    return new

def exp_def(x, **kwargs):
    """ Apply exp on x """
    new = np.array([np.exp(x)])
    return new

def cos_def(x, **kwargs):
    """ Apply cos on x """
    new = np.array([np.cos(x)])
    return new

def log_def(x, **kwargs):
    """ Apply log on x """
    new = np.array([np.log(x)])
    return new

def multiply(x, **kwargs):
    """polynomial basis function."""
    degree = kwargs.get('degree')
    new = np.array([ x**degree ])
    return new

def build_poly(x, **kwargs):
    """polynomial basis function."""
    degree = kwargs.get('degree')
    new = np.array([ [ e**i for e in x ] for i in range(1,degree+1) ])
    return new.T

def build_poly_matrix(tx, degree):
    """ apply build_poly to all columns of tx """
    res = [ build_poly(x, degree=degree) for x in tx.T ]
    conc = np.concatenate(res, axis=1)
    one = np.ones((tx.shape[0], 1))
    return np.concatenate([one, conc], axis=1)
