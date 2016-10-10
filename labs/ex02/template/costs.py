# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_cost(y, tx, w):
    """calculate the cost.

    you can calculate the cost by mse or mae.
    """
    N = y.shape[0]
    e = y-np.dot(tx,w)
    return 1/(2*N)*np.dot(e.T,e)
