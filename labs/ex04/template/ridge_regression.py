# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    M = tx.shape[1]
    N = len(y)
    big_lamb = lamb*2*N
    #w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx) + big_lamb*np.eye(M)), tx.T),y)
    w = np.linalg.solve(np.dot(tx.T,tx) + big_lamb*np.eye(M), np.dot(tx.T, y))
    return w
