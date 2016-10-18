# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis function."""
    new = np.array([ [ e**i for e in x ] for i in range(degree+1) ])
    return new.T

def build_poly_matrix(tx, degree):
    res = [ build_poly(x, degree) for x in tx.T ]
    return np.concatenate(res, axis=1)

"""def build_poly(x, degree):
    polynomial basis functions for input data x, for j=0 up to j=degree.    
    Phi_tilde = np.ones((len(x),1))
    for i in range(1, degree+1):
        power_column = []
        for j in range (len(x)):
            power_column = np.append(power_column, np.power(x[j],i))
   
        print(np.shape(Phi_tilde))
        print(len(x))
        print(np.shape([len(x),1]))
        Phi_tilde = np.concatenate((Phi_tilde,power_column.reshape([len(x),1])), axis=1)
    
    return Phi_tilde_bis"""