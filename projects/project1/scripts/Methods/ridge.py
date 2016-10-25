# -*- coding: utf-8 -*-

import numpy as np
from Methods.costs import * 

def ridge_regression(y, tx, **kwargs):
    """implement ridge regression."""
    lamb = kwargs.get('lamb')
    M = tx.shape[1]
    N = len(y)
    big_lamb = lamb*2*N
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx) + big_lamb*np.eye(M)), tx.T),y)
    return w