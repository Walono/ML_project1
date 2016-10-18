# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    x = x.reshape([len(x),1])
    y = y.reshape([len(y),1])
    xy = np.concatenate((x, y), axis=1)
    random_xy = np.random.shuffle(xy)
    x = xy[:,0]
    y = xy[:,1]
    
    ratio = np.rint(len(x)*ratio)
    
    x_tr = x[0:ratio]
    x_te = x[ratio:len(x)]
    y_tr = y[0:ratio]
    y_te = y[ratio:len(y)]
    
    return x_tr, y_tr, x_te, y_te
