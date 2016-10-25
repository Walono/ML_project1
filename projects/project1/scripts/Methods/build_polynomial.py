# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def add_feature(y, x, tX, method, **kwargs):
    prev_corr = abs(np.corrcoef(y,x)[1,0])
    x_transform = method(x, **kwargs)
    new_corr = abs(np.corrcoef(y,x_transform[0])[1,0])
    newtX = tX
    if new_corr > prev_corr :
        newtX = np.vstack((newtX.T, x_transform[0]))
        newtX = newtX.T
    return newtX

def log_def(x, **kwargs):
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
    res = [ build_poly(x, degree=degree) for x in tx.T ]
    conc = np.concatenate(res, axis=1)
    one = np.ones((tx.shape[0], 1))
    return np.concatenate([one, conc], axis=1)

"""def build_poly(x, degree):
    polynomial basis functions for input data x, for j=0 up to j=degree.    
    Phi_tilde = np.ones((len(x),1))
    for i in range(1, degree+1):
        power_column = []
        for j in range (len(x)):
            power_column = np.append(power_column, np.power(x[j],i))
        Phi_tilde = np.concatenate((Phi_tilde,power_column.reshape([len(x),1])), axis=1)
    
    return Phi_tilde_bis"""
