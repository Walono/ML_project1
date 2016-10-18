from Methods.costs import *
from Methods.ridge import *
from Methods.build_polynomial import *
from Methods.plots import *


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    k_fold = list(range(k_indices.shape[0]))
    k_fold.remove(k)
    indices_tr = k_indices[k_fold].ravel()
    y_tr = y[indices_tr]
    x_tr = x[indices_tr]
    y_te = y[k_indices[k]]
    x_te = x[k_indices[k]]
    
    Phi_tr = build_poly(x_tr, degree)
    Phi_te = build_poly(x_te, degree)
   
    w_tr = ridge_regression(y_tr, Phi_tr, lambda_)
    loss_tr = compute_mse(y_tr, Phi_tr, w_tr)
    loss_te = compute_mse(y_te, Phi_te, w_tr)
    
    return loss_tr, loss_te


def cross_validation_demo():
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
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lamb, degree)
            tot_loss_tr += loss_tr
            tot_loss_te += loss_te
            #print(tot_loss_tr, tot_loss_te)
        rmse_tr.append(np.sqrt(2/k_fold * tot_loss_tr))
        rmse_te.append(np.sqrt(2/k_fold * tot_loss_te))

    plt.boxplot(rmse_te)
    plt.show()
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)

