# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""    
    Phi_tilde = np.ones((len(x),1))
    for i in range(1, degree+1):
        power_column = []
        for j in range (len(x)):
            power_column = np.append(power_column, np.power(x[j],i))
            #Phi.append(x[j]**i)
        #print(power_column.shape)    
        #np.concatenate((Phi, power_column), axis=1)
        Phi_tilde = np.concatenate((Phi_tilde,power_column.reshape([len(x),1])), axis=1)
    
    return Phi_tilde