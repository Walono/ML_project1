# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import Methods.costs


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    
    return -1/N*np.dot(tx.T, e)


def gradient_descent(y, tx, **kwargs): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    initial_w = kwargs.get('initial_w')
    max_iters = kwargs.get('max_iters')
    gamma = kwargs.get('gamma')
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_cost(y, tx, w)
        gradient = compute_gradient(y, tx, w)

        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws
