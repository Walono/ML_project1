# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    
    return -1/N*np.dot(tx.T, e)


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_epochs):
        batch_iterator = batch_iter(y, tx, batch_size)
        batch_y, batch_tx = next(batch_iterator)
        loss = compute_cost(batch_y, batch_tx, w)
        gradient = compute_stoch_gradient(batch_y, batch_tx, w)

        w = w - gamma*gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return losses, ws
