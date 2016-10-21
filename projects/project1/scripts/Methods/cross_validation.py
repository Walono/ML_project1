from Methods.costs import *
from Methods.ridge import *
from Methods.least_squares import *
from Methods.build_polynomial import *
from Methods.plots import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    k_fold = list(range(k_indices.shape[0]))
    k_fold.remove(k)
    indices_tr = k_indices[k_fold].ravel()
    y_tr = y[indices_tr]
    x_tr = x[indices_tr]
    y_te = y[k_indices[k]]
    x_te = x[k_indices[k]]
   
    w_tr = ridge_regression(y_tr, x_tr, lambda_)
    loss_tr = compute_mse(y_tr, x_tr, w_tr)
    loss_te = compute_mse(y_te, x_te, w_tr)
    return loss_tr, loss_te



def cross_validation_demo(y,tX):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 2, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    k_list = list(range(k_fold))
    for lamb in lambdas:
        tot_loss_tr = 0
        tot_loss_te = 0
        for k in k_list:
            loss_tr, loss_te = cross_validation(y, tX, k_indices, k, lamb)
            tot_loss_tr += loss_tr
            tot_loss_te += loss_te
        rmse_tr.append(np.sqrt(2/k_fold * tot_loss_tr))
        rmse_te.append(np.sqrt(2/k_fold * tot_loss_te))
    plt.boxplot(rmse_te)
    plt.show()
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
