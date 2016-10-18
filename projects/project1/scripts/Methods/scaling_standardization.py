# -*- coding: utf-8 -*-

import numpy as np

# DATA SCALING method:

def data_scaling(x):
    minma_x = (x - x.min()) / (x.max() - x.min())
    return(minma_x.T)

# DATA STANDARDIZATION METHOD:

def data_standardization(x):
    centered_x = x - np.mean(x, axis=0)
    std_x = centered_x / np.std(centered_x, axis=0)

    return(std_x.T)