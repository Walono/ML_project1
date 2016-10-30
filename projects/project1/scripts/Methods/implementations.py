import numpy as np
from Methods.costs import * 


least squares GD(y, tx, initial w, max iters, gamma)

least squares SGD(y, tx, initial w, max iters, gamma)

least squares(y, tx)

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    M = tx.shape[1]
    N = len(y)
    big_lamb = lamb*2*N
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx) + big_lamb*np.eye(M)), tx.T),y)
    return w

logistic regression(y, tx, initial w, max iters, gamma)

reg logistic regression(y, tx, lambda , initial w, max iters, gamma)



def least_squares(y, tx):
    w = np.dot(np.dot(np.linalg.inv(np.dot(
        np.transpose(tx),tx)),np.transpose(tx)),y)
    mse = compute_loss(y, tx, w)
    return mse, w
	
	
