from Methods.costs import *
from Methods.ridge import *
from Methods.least_squares import *
from Methods.logistic import *
from Methods.build_polynomial import *
from Methods.plots import *
from Methods.proj1_helpers import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, model, **kwargs):
    k_fold = list(range(k_indices.shape[0]))
    k_fold.remove(k)
    indices_tr = k_indices[k_fold].ravel()
    y_tr = y[indices_tr]
    x_tr = x[indices_tr]
    y_te = y[k_indices[k]]
    x_te = x[k_indices[k]]
   
    if 'lambda_' in kwargs:
        w_tr = model(y_tr, x_tr, kwargs.get('lambda_'))
    else:
        w_tr = model(y_tr, x_tr)
    
    prediction = np.dot(w_tr,x_te.T)
    
    #print(prediction)
    prediction[prediction < 0] = -1
    prediction[prediction >= 0] = 1

    #print(prediction)
    #print(y_te)
    accuracy = np.sum(y_te == prediction) / float(len(y_te))
    
    
    #m, w_tr = least_squares(y_tr, x_tr)
    e_tr = y_tr - predict_labels(w_tr, x_tr)
    e_te = y_te - predict_labels(w_tr, x_te)
    loss_tr = calculate_mse(e_tr)
    loss_te = calculate_mse(e_te)
    #loss_tr = compute_mse(y_tr, x_tr, w_tr)
    #loss_te = compute_mse(y_te, x_te, w_tr)
    return loss_tr, loss_te, accuracy, w_tr

