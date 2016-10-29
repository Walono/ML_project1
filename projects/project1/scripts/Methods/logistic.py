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

def logistic_regression_gradient_descent(y, tx, gamma, max_iters):
    # init parameters
    threshold = 1e-8
    batch_size = 3000
    losses = []

    w = np.zeros((tx.shape[1], 1))
    #w = np.array([w]).T

    # start the logistic regression
    for iter in range(max_iters):
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
    print("The scaled loss={l}".format(l=calculate_loss(y, tx, w)))
    return w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    loss = calculate_loss(y, tx, w) + lambda_ * np.dot(w.T, w) / len(y)
    grad = calculate_gradient(y, tx, w) + lambda_ * w[:,0]
    hess = calculate_hessian(y, tx, w) + lambda_ * np.identity(tx.shape[1])
    # return loss, gradient, and hessian
    # ***************************************************
    return loss, grad, hess

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient and hessian: TODO
    # ***************************************************
    loss, grad, hess = penalized_logistic_regression(y, tx, w, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    hess_inv = np.linalg.inv(hess)
    w = w - gamma * np.array([np.dot(hess_inv, grad)]).T
    #w = w - gamma * np.array([grad]).T
    return loss, w

def logistic_regression_penalized_gradient_descent(y, tx, gamma, max_iters, lambda_):
    # init parameters
    threshold = 1e-8
    losses = []
    batch_size = 3000

    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iters):
        batch_iterator = batch_iter(y, tx, batch_size)
        batch_y, batch_tx = next(batch_iterator)
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(batch_y, batch_tx, w, gamma, lambda_)
        # log info
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
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
