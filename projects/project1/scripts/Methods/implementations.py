import numpy as np
import matplotlib.pyplot as plt
from costs import *


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    
    return -1/N*np.dot(tx.T, e)


def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma*gradient
        loss = compute_loss(y, tx, w)

    return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    
    return -1/N*np.dot(tx.T, e)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    batch_size = 1
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        batch_iterator = batch_iter(y, tx, batch_size)
        batch_y, batch_tx = next(batch_iterator)
        gradient = compute_stoch_gradient(batch_y, batch_tx, w)

        w = w - gamma*gradient
        loss = compute_loss(batch_y, batch_tx, w)

    return w, loss

def least_squares(y, tx):
    w = np.dot(np.dot(np.linalg.inv(np.dot(
        np.transpose(tx),tx)),np.transpose(tx)),y)
    mse = compute_loss(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    M = tx.shape[1]
    N = len(y)
    big_lamb = lambda_*2*N
    w = np.dot(np.dot(np.linalg.inv(np.dot(tx.T,tx) + big_lamb*np.eye(M)), tx.T),y)
    loss = compute_loss(y, tx, w)
    return w, loss


def sigmoid(t):
    """apply sigmoid function on t."""
    sig = 1 / (1 + np.exp(-t))
    return sig


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    N = len(y)
    loss = 0
    for n in range(N):
        l = np.log(1 + np.exp(np.dot(tx[n], w)))
        m = y[n] * np.dot(tx[n], w)
        loss += l[0] - m[0]

    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss function."""
    sig = sigmoid(np.dot(tx, w))
    temp = sig[:,0] - y
    grad = np.dot(tx.T, temp)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    grad = calculate_gradient(y, tx, w)
    w = w - gamma * np.array([grad]).T

    loss = calculate_loss(y, tx, w)
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    batch_size = 1
    losses = []
    
    if np.array(initial_w).ndim == 1:
        w = np.array([initial_w]).T
    else:
        w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        batch_iterator = batch_iter(y, tx, batch_size)
        batch_y, batch_tx = next(batch_iterator)
        loss, w = learning_by_gradient_descent(batch_y, batch_tx, w, gamma)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return w, loss

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    loss = calculate_loss(y, tx, w) + lambda_ * np.dot(w.T, w)[0,0]
    grad = calculate_gradient(y, tx, w) + lambda_ * w[:,0]
    # return loss, gradient
    # ***************************************************
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)

    w = w - gamma * np.array([grad]).T
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []
    batch_size = 1
        
    if np.array(initial_w).ndim == 1:
        w = np.array([initial_w]).T
    else:
        w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        batch_iterator = batch_iter(y, tx, batch_size)
        batch_y, batch_tx = next(batch_iterator)
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(batch_y, batch_tx, w, gamma, lambda_)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, calculate_loss(y, tx, w)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
