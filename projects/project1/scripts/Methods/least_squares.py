# -*- coding: utf-8 -*-

import numpy as np
from Methods.costs import * 


def least_squares(y, tx):
    """ Compute least_squares """
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T, y))
    mse = compute_loss(y, tx, w)
    return w
