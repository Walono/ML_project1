# -*- coding: utf-8 -*-

import numpy as np
from Methods.costs import * 

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    M = tx.shape[1]
    N = len(y)
    big_lamb = lambda_*2*N
    w = np.linalg.solve(np.dot(tx.T,tx) + big_lamb*np.eye(M), np.dot(tx.T, y))
    return w