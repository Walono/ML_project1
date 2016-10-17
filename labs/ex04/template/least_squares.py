# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse

def least_squares(y, tx):
    """calculate the least squares."""
    #w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx),tx)),np.transpose(tx)),y)
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T, y))
    mse = compute_mse(y, tx, w)
    return mse, w
