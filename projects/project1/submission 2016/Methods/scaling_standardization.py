# -*- coding: utf-8 -*-

import numpy as np

# DATA SCALING method:

def data_scaling(x, x_te=None):
    minima_x = (x - x.min()) / (x.max() - x.min())
    if x_te is not None:
        minima_x_te = (x_te - x.min()) / (x.max() - x.min())
        return minima_x, minima_x_te
    return minima_x

# DATA STANDARDIZATION METHOD:

def data_standardization(x):
    centered_x = x - np.mean(x, axis=0)
    std_x = centered_x / np.std(centered_x, axis=0)

    return(std_x.T)