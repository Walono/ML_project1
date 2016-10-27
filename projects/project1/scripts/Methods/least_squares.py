# -*- coding: utf-8 -*-

import numpy as np
from Methods.costs import * 


def least_squares(y, tx):
    #w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx),tx)),np.transpose(tx)),y)
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T, y))
    mse = compute_loss(y, tx, w)
    return w