# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    y = y.reshape([len(y),1])
    xy = np.concatenate((x, y), axis=1)
    random_xy = np.random.shuffle(xy)
    x = xy[:,0:xy.shape[1]-1]
    y = xy[:,xy.shape[1]-1]
    #ratio = np.rint(len(x)*ratio)
    ratio = np.round(len(x)*ratio).astype(np.int)
    x_tr = x[0:ratio]
    x_te = x[ratio:len(x)]
    y_tr = y[0:ratio]
    y_te = y[ratio:len(y)]
    
    return x_tr, y_tr, x_te, y_te

'''
def split_data(tx, y, ratio, seed=1):
    split the dataset based on the split ratio.
    # set seed
    np.random.seed(seed)
    y = y.reshape([len(y),1])
    for x in tx:
        x = x.reshape([len(x),1])
    txy = np.concatenate((tx, y), axis=1)
    random_xy = np.random.shuffle(txy)
    print(len(txy[1,:]))
    shuffle_tx = np.array([])
    for i in range(0,len(txy[1,:])):
        shuffle_tx.np.append(txy[:,i])
    y = txy[:,len(txy[1,:])]
    
    ratio = np.rint(len(x)*ratio)
    
    x_tr = x[0:ratio]
    x_te = x[ratio:len(x)]
    y_tr = y[0:ratio]
    y_te = y[ratio:len(y)]
    
    return x_tr, y_tr, x_te, y_te
'''