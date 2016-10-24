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
    return loss

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

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    #loss = compute_loss(y, tx, w[:,0])

    grad = calculate_gradient(y, tx, w)

    w = w - gamma * np.array([grad]).T
    return loss, w

def logistic_regression_gradient_descent_demo(y, tx):
    # init parameters
    max_iter = 4000
    threshold = 1e-8
    gamma = 0.0000001
    losses = []

    w = np.zeros((tx.shape[1], 1))
    #w = np.array([w]).T
    print(tx.shape[1], w.shape)

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 5 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return w
