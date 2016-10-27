import numpy as np
import matplotlib.pyplot as plt
from Methods.proj1_helpers import *
from Methods.costs import *

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
    #log_array = np.array([ np.log(1 + np.exp(np.dot(tx[n], w))) - y*np.dot(tx[n], w) for n in range(N)])
    #cost = np.sum(log_array)
    return loss/N

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig = sigmoid(np.dot(tx, w))
    temp = sig[:,0] - y
    grad = np.dot(tx.T, temp)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    return grad

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    N = len(y)
    S = np.zeros((N, N))
    for i in range(N):
        prod = sigmoid(np.dot(tx[i], w))[0]
        S[i, i] = prod * (1 - prod)
    h_temp = np.dot(S, tx)
    H = np.dot(tx.T, h_temp)
    #print(H, "PLOP")
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    # ***************************************************
    return H

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    #loss = compute_loss(y, tx, w[:,0])

    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    hess_inv = np.linalg.inv(hess)
    #w = w - gamma * np.array([grad]).T
    w = w - gamma * np.array([np.dot(hess_inv, grad)]).T
    return loss, w

def logistic_regression_gradient_descent_demo(y, tx, **kwargs):
    # init parameters
    max_iter = 2000
    threshold = 1e-8
    gamma = 0.0003
    batch_size = kwargs['batch_size']
    losses = []

    w = np.zeros((tx.shape[1], 1))
    #w = np.array([w]).T
    print(tx.shape[1], w.shape)

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        batch_iterator = batch_iter(y, tx, batch_size)
        batch_y, batch_tx = next(batch_iterator)
        loss, w = learning_by_gradient_descent(batch_y, batch_tx, w, gamma)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return w

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
